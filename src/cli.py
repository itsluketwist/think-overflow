"""CLI entry point for think-overflow inference."""

from argparse import ArgumentParser

from src.run_inference import _OVERFLOW_SUFFIXES, run_inference


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
    default="greedymax",
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
    "--onepass",
    action="store_true",
    default=False,
    dest="onepass",
    help=(
        "Run onepass unconstrained inference instead of two-pass inference. "
        "Mutually exclusive with --max-think-tokens."
    ),
)

parser.add_argument(
    "--max-think-tokens",
    type=int,
    default=None,
    help="Token cap for Pass 1 reasoning. Suggested sweep: 2048, 4096, 8192, 16384.",
)

parser.add_argument(
    "--overflow-suffix",
    type=str,
    choices=list(_OVERFLOW_SUFFIXES.keys()),
    default="formal",
    help=(
        "Suffix appended to truncated reasoning before closing </think>. "
        "base=nothing (pure truncation), truncated=' [reasoning truncated]', "
        "formal='... I have to stop thinking and answer now.', "
        "human='... oops, I really need to stop thinking and to answer.'"
    ),
)

parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Short tag appended to output filenames. Defaults to 'mt{max_think_tokens}'.",
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

    # validate: exactly one of --onepass or --max-think-tokens must be provided
    if args.onepass and args.max_think_tokens is not None:
        parser.error("--onepass and --max-think-tokens are mutually exclusive.")
    if not args.onepass and args.max_think_tokens is None:
        parser.error("one of --onepass or --max-think-tokens is required.")

    run_inference(
        model=args.model,
        model_config=args.model_config,
        config_file=args.config_file,
        config_profile=args.config_profile,
        output=args.output,
        datasets=args.datasets,
        eval_type=args.type,
        onepass=args.onepass,
        max_think_tokens=args.max_think_tokens,
        overflow_suffix=args.overflow_suffix,
        run_name=args.run_name,
        debug=args.debug,
        update=args.update,
    )
