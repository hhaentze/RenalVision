from renal_vision.features import cli as feature_cli
from renal_vision.modeling import cli as model_cli
from renal_vision.shared.parser_config import get_base_parser


def main():
    parser = get_base_parser()

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
