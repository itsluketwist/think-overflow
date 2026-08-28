"""CLI entry point for think-overflow inference."""

from argparse import ArgumentParser

from src.run_inference import _OVERFLOW_SUFFIXES, _PROMPT_SUFFIXES, run_inference


parser = ArgumentParser(
    description="Two-pass capped-reasoning inference for the think-overflow method.",
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help="Model key from models.yaml.",
)

parser.add_argument(
    "-mc",
    "--model-config",
    type=str,
    default="config/models.yaml",
    help="Path to model config file.",
)

parser.add_argument(
    "-cf",
    "--config-file",
    type=str,
    default="config/inference.yaml",
    help="Path to inference config file.",
)

parser.add_argument(
    "-cp",
    "--config-profile",
    type=str,
    default="greedy",
    help="Config profile name within the config file.",
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="output/",
    help="Base directory for outputs.",
)

parser.add_argument(
    "-d",
    "--datasets",
    type=str,
    required=True,
    help="Comma-separated datasets. Use 'path' for auto-detected type or 'path:type' for explicit.",
)

parser.add_argument(
    "-t",
    "--type",
    type=str,
    default=None,
    help="Evaluation type fallback for unknown datasets (e.g. 'reasoning').",
)

parser.add_argument(
    "--max-tokens",
    type=int,
    default=None,
    help="Overall token budget: total generation tokens for onepass, or combined Pass 1 + Pass 2 tokens for twopass.",
)

parser.add_argument(
    "--max-think-tokens",
    type=int,
    default=None,
    help=(
        "Token cap for Pass 1 reasoning. Suggested sweep: 2048, 4096, 8192, 16384. "
        "Use 0 to run nothink mode: injects an empty <think></think> block and skips "
        "reasoning entirely, forcing the model to answer directly."
    ),
)

parser.add_argument(
    "--overflow-suffix",
    type=str,
    choices=list(_OVERFLOW_SUFFIXES.keys()),
    default="formal",
    help=(
        "Suffix appended to truncated reasoning before closing </think>. "
        "base=nothing (pure truncation), dots='...', "
        "truncated=' [reasoning truncated]', "
        "formal='... I have to stop thinking and answer now.', "
        "human='... oops, I really need to stop thinking and to answer.'"
    ),
)

parser.add_argument(
    "--prompt-suffix",
    type=str,
    choices=list(_PROMPT_SUFFIXES.keys()),
    default="none",
    help=(
        "Instruction appended to the end of every user prompt (onepass only). "
        "Used to test whether prompt-engineering shortens reasoning. "
        "none=nothing (baseline). Combining with --max-think-tokens is an error."
    ),
)

parser.add_argument(
    "--skip-ids",
    type=str,
    default=None,
    help=(
        "Comma-separated task_id values to skip during code evaluation, for tasks "
        "whose generated code is unsafe to execute (e.g. 'BigCodeBench/348' asks "
        "for process-killing code that can terminate the evaluation itself). "
        "Skipped tasks are excluded from all metrics."
    ),
)

parser.add_argument(
    "-u",
    "--update",
    action="store_true",
    default=False,
    help="Force regeneration of responses even if the output file already exists.",
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help=(
        "Debug mode: limit to the first 5 samples from one dataset per eval type, "
        "and save all output to output/debug/."
    ),
)


def main() -> None:
    """Parse and validate CLI args, then dispatch to run_inference."""
    args = parser.parse_args()

    run_inference(
        model=args.model,
        model_config=args.model_config,
        config_file=args.config_file,
        config_profile=args.config_profile,
        output=args.output,
        datasets=args.datasets,
        eval_type=args.type,
        max_think_tokens=args.max_think_tokens,
        max_tokens=args.max_tokens,
        overflow_suffix=args.overflow_suffix,
        prompt_suffix=args.prompt_suffix,
        skip_ids=args.skip_ids,
        debug=args.debug,
        update=args.update,
    )
