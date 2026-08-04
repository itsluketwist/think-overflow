"""Evaluation of code responses by executing tests."""

import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
from typing import Any

from llm_cgr import Markdown
from thinkpack import ParsedResponse

from src.evaluate.statistics import compute_pass_at_1, wilson_ci


# default timeout for standard code execution
_DEFAULT_TIMEOUT_SECONDS = 5.0
# bigcodebench tasks import heavy libraries (pandas, numpy) so need more time
_BCB_TIMEOUT_SECONDS = 30.0
_DEFAULT_MEMORY_MB = 512


def _extract_code(response: str) -> str:
    """Extract code from markdown blocks if present; return the response as-is otherwise."""
    _markdown = Markdown(text=response, default_codeblock_language="python")

    _code = [
        _cb.text
        for _cb in _markdown.code_blocks
        if _cb.language == "python" and _cb.valid
    ]

    if _code:
        return max(_code, key=len)
    return response


def _is_unittest_test(test_code: str) -> bool:
    """Return True if test_code uses unittest.TestCase format (e.g. BigCodeBench)."""
    return "unittest" in test_code


def _set_memory_limit(memory_mb: int) -> None:
    """Set the virtual memory limit for the current process (child only via preexec_fn).

    Skipped on macOS — RLIMIT_AS is defined but enforcing it crashes Python subprocesses.
    """
    if platform.system() == "Darwin":
        # macOS: skip — RLIMIT_AS crashes Python subprocesses at startup
        return
    try:
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        # platform does not support RLIMIT_AS or limit is invalid
        pass


def _execute_code(
    code: str,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = _DEFAULT_MEMORY_MB,
    python_executable: str | None = None,
) -> tuple[bool, str]:
    """Run code in a sandboxed subprocess with time and memory limits.

    Pass python_executable to use a different interpreter (e.g. a venv with
    extra libraries installed); defaults to the current interpreter.

    Returns (success, output/error message).
    """
    executable = python_executable or sys.executable
    try:
        # run in a temp dir so any files created by the code don't pollute the repo
        # on slurm, $TMPDIR points to fast per-job scratch storage
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [executable, "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=lambda: _set_memory_limit(memory_mb),
                cwd=tmp_dir,
            )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds}s"

    # success: return captured stdout
    if result.returncode == 0:
        return True, result.stdout

    # killed by a signal (negative return code on unix)
    if result.returncode < 0:
        sig_num = -result.returncode
        try:
            sig_name = signal.Signals(sig_num).name
        except (ValueError, AttributeError):
            sig_name = f"signal {sig_num}"
        return False, f"killed by {sig_name}"

    # normal failure — use stderr if available, otherwise stdout
    error = result.stderr.strip() or result.stdout.strip()
    return False, error or f"exit code {result.returncode}"


def _build_test_code(
    code: str,
    record: dict[str, Any],
) -> str:
    """Build executable test code from extracted code and a record.

    Supports: (1) test_code only, (2) inputs/expected_outputs pairs,
    (3) unittest.TestCase style (bigcodebench), (4) entry_point + test_code.

    Returns the complete Python code to execute.
    """
    # format 1: direct test_code field (no entry_point)
    # the test_code should reference the generated code directly
    if "test_code" in record and "entry_point" not in record:
        return f"{code}\n\n{record['test_code']}"

    # format 2: input/output pairs (competitive programming style)
    # wrap code in a function and run it for each test case
    if "inputs" in record and "expected_outputs" in record:
        # escape the code for embedding in a string
        escaped_code = code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

        test_cases = list(zip(record["inputs"], record["expected_outputs"]))
        test_case_strs = [
            f"({repr(inp)}, {repr(exp.strip() if isinstance(exp, str) else exp)})"
            for inp, exp in test_cases
        ]

        return f'''
import sys
from io import StringIO

_SOLUTION_CODE = """{escaped_code}"""

def _run_solution(stdin_input: str) -> str:
    """Run the solution code with mocked stdin and return stdout."""
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = StringIO(stdin_input)
    captured = StringIO()
    sys.stdout = captured

    try:
        exec(_SOLUTION_CODE, {{"__name__": "__main__"}})
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout

    return captured.getvalue().strip()

# run all test cases
_test_cases = [{", ".join(test_case_strs)}]
for _idx, (_input, _expected) in enumerate(_test_cases):
    _output = _run_solution(_input)
    assert _output == _expected, f"test {{_idx + 1}}: expected {{_expected!r}}, got {{_output!r}}"
'''

    # format 3: unittest.TestCase style (bigcodebench) — checked before format 4
    # since bcb records also have entry_point + test_code
    if "test_code" in record and _is_unittest_test(record["test_code"]):
        # append unittest runner; exit code 0 = all pass, 1 = any failure
        return (
            f"{code}\n\n{record['test_code']}\n\nunittest.main(exit=True, verbosity=0)"
        )

    # format 4: entry_point + test_code (evalplus / humaneval style)
    # the code defines a function, test_code calls it via assertions
    if "entry_point" in record and "test_code" in record:
        return f"{code}\n\n{record['test_code']}"

    # fallback: just try to execute the code with any test_code
    if "test_code" in record:
        return f"{code}\n\n{record['test_code']}"

    # no tests provided — just check if code executes without error
    return code


def _evaluate_response(
    response: str,
    record: dict[str, Any],
    python_executable: str | None = None,
) -> tuple[bool, str | None]:
    """Build and execute test code from a response against a record's test cases.

    Pass python_executable to run tests under a different interpreter (e.g. a
    dedicated venv with benchmark-specific library versions installed).

    Returns (passed, output).
    """
    code = _extract_code(response)
    test_code = _build_test_code(
        code=code,
        record=record,
    )
    # use a longer timeout for unittest-style tests (bigcodebench) due to heavy imports
    timeout = (
        _BCB_TIMEOUT_SECONDS
        if _is_unittest_test(record.get("test_code", ""))
        else _DEFAULT_TIMEOUT_SECONDS
    )
    success, output = _execute_code(
        code=test_code,
        timeout_seconds=timeout,
        python_executable=python_executable,
    )
    return success, output


def evaluate_code(
    responses: list[list[ParsedResponse]],
    records: list[dict[str, Any]],
    dataset_name: str | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate code responses by executing them against test cases in each record.

    Evaluates the answer section only; also checks correct_in_reasoning as a diagnostic.

    BigCodeBench requires a dedicated venv with pinned libraries (numpy, pandas, scipy,
    etc.). Set the BCB_PYTHON environment variable to the path of that venv's interpreter;
    if unset, the current interpreter is used and imports may fail.

    Returns a dict with pass_at_1, pass_at_1_list, pass_at_k, k, total, and a per-task breakdown.
    """
    results = []
    k = len(responses[0]) if responses and responses[0] else 1
    # track passed count per sample index for averaged pass@1 calculation
    per_sample_passed = [0] * k
    pass_at_k_count = 0

    # bigcodebench requires its own venv for pinned library versions; read path from env
    python_executable: str | None = (
        os.environ.get("BCB_PYTHON") if dataset_name == "bigcodebench" else None
    )

    for sample_parsed, record in zip(responses, records):
        sample_results: list[dict[str, Any]] = []

        for p in sample_parsed:
            # primary evaluation: answer section only
            # empty answer section = model failed to produce code after reasoning
            if p.answer.strip():
                success, output = _evaluate_response(
                    response=p.answer,
                    record=record,
                    python_executable=python_executable,
                )
            else:
                success, output = False, None

            # diagnostic: check if passing code appeared in the reasoning block
            # (less common but possible, especially for olmo3-style models)
            correct_in_reasoning: bool | None = None
            if p.reasoning.strip():
                reasoning_success, _ = _evaluate_response(
                    response=p.reasoning,
                    record=record,
                    python_executable=python_executable,
                )
                correct_in_reasoning = reasoning_success

            sample_results.append(
                {
                    "passed": success,
                    "output": output[:500] if output else None,  # truncate long output
                    "answer_extracted": p.answer.strip() != "",
                    "correct_in_reasoning": correct_in_reasoning,
                }
            )

        # track passed per sample index — used to compute averaged pass@1 below
        # skip indices beyond k: later tasks may have more samples than the first
        for i, r in enumerate(sample_results):
            if i >= k:
                break
            if r["passed"]:
                per_sample_passed[i] += 1

        # pass@k: any sample passes
        any_passed = any(r["passed"] for r in sample_results)
        if any_passed:
            pass_at_k_count += 1

        # per-task pass@1 stored as first-sample bool (used in breakdown only)
        first_passed = sample_results[0]["passed"] if sample_results else False
        results.append(
            {
                "samples": sample_results,
                "pass_at_1": first_passed,
                "pass_at_k": any_passed,
            }
        )

    total = len(results)
    # averaged pass@1: treat each sample index as an independent trial and average
    pass_at_1_list, pass_at_1, pass_at_1_ci = compute_pass_at_1(
        counts=per_sample_passed,
        total=total,
        k=k,
    )
    pass_at_k_ci = wilson_ci(count=pass_at_k_count, total=total)

    return {
        "pass_at_1": pass_at_1,
        "pass_at_1_ci": pass_at_1_ci,
        "pass_at_1_list": pass_at_1_list,
        "pass_at_k": pass_at_k_count / total if total else 0.0,
        "pass_at_k_ci": pass_at_k_ci,
        "k": k,
        "total": total,
    }, results
