"""Dataset loading and prompt construction utilities."""

from pathlib import Path

from datasets import load_dataset
from llm_cgr import load_jsonl


# instruction appended to prompts for math evaluation
_BOXED_INSTRUCTION = "\n\nPlease provide your final answer in a \\boxed{} environment."

# valid evaluation types (used for auto-detection from directory structure)
EVAL_TYPES = {"math", "code", "reasoning", "crux"}


def load_dataset_records(
    dataset: str,
    default_dir: str,
    split: str,
    required_keys: set[str] | None = None,
) -> list[dict[str, str]]:
    """Load dataset records from various sources.

    Supports three formats:
    - Local path: if the path exists, loads directly
    - Simple name: loads from {default_dir}/{name}.jsonl
    - HuggingFace ID: loads from HuggingFace Hub (e.g. 'org/dataset-name')

    Optionally validates that each record contains the required keys.

    Returns a list of records.
    """
    records: list[dict[str, str]] = []

    # check if it's a local file path
    local_path = Path(dataset)
    if local_path.exists():
        records = load_jsonl(file_path=str(local_path))

    # check if it resolves to a file in the default directory
    elif (default_path := Path(default_dir) / f"{dataset}.jsonl").exists():
        records = load_jsonl(file_path=str(default_path))

    # otherwise, treat as a huggingface dataset id
    else:
        hf_dataset = load_dataset(dataset, split=split)
        records = [dict(record) for record in hf_dataset]

    # validate required keys if specified
    if required_keys:
        for i, record in enumerate(records):
            missing = required_keys - set(record.keys())
            if missing:
                raise ValueError(
                    f"record {i} is missing required keys: {missing}",
                )

    return records


def parse_dataset_entry(entry: str) -> tuple[str, str | None]:
    """Parse a dataset entry string and determine its evaluation type.

    Supports formats:
    - "type/dataset" -> auto-detect type from directory (e.g., 'math/gsm8k' -> 'math')
    - "path:type" -> explicit type (e.g., 'custom/data:reasoning')

    Returns tuple of (dataset_path, eval_type).
    """
    # check for explicit type suffix (e.g., "path:reasoning")
    if ":" in entry:
        path, eval_type = entry.rsplit(":", 1)
        return path, eval_type

    # auto-detect type from parent directory (e.g., "math/gsm8k" -> "math")
    parts = Path(entry).parts
    if len(parts) >= 2:
        parent = parts[-2].lower()
        if parent in EVAL_TYPES:
            return entry, parent

    return entry, None


def construct_inference_prompts(
    records: list[dict],
    eval_type: str | None = None,
    prompt_suffix: str | None = None,
) -> list[str]:
    """Construct prompts for inference from dataset records.

    Extracts the 'prompt' field from each record and applies any
    eval-type-specific modifications (e.g., boxed instruction for math).
    If prompt_suffix is provided, it is appended to every prompt after
    any eval-type modification.

    Returns a list of prompt strings.
    """
    prompts = [record["prompt"] for record in records]

    # for math, append instruction to put answer in \boxed{}
    if eval_type == "math":
        prompts = [p + _BOXED_INSTRUCTION for p in prompts]

    # append strategy-level suffix to the user prompt if provided
    if prompt_suffix is not None:
        prompts = [p + prompt_suffix for p in prompts]

    return prompts
