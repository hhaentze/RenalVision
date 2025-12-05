import argparse

from renal_vision.features import cli as feature_cli
from renal_vision.modeling import cli as model_cli


def main():
    parser = argparse.ArgumentParser(
        description="RenalVision Central CLI", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create the shared subparser registry
    subparsers = parser.add_subparsers(dest="command", required=True)
    model_cli.add_subparsers(subparsers)
    feature_cli.add_subparsers(subparsers)

    # --- Execution ---
    args = parser.parse_args()

    # This single line replaces the entire if/else chain
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
