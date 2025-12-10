"""
Main command-line interface for the Cyst Classifier.
"""

import argparse

from renal_vision.shared.parser_config import get_base_parser


# ================= Logic =================
def run_train(args: argparse.Namespace) -> None:
    from . import train

    train.run_training(
        data_path=args.data,
        output_dir=args.output_dir,
        model_type=args.model,
        class_config_path=args.class_config,
        extractor_config_path=args.extractor_config,
        tree_max_depth=args.depth,
    )


def run_eval(args: argparse.Namespace) -> None:
    from . import eval

    eval.run_evaluation(
        data_path=args.data,
        model_path=args.model,
        output_dir=args.output_dir,
    )


def run_infer(args: argparse.Namespace) -> None:
    from . import inference

    predictor = inference.LesionPredictor(model_identifier=args.model)

    if args.output:
        # Multi-lesion / file output mode
        predictor.infer_mask(args.image, args.seg, args.output)
    else:
        # Single lesion / print mode
        try:
            result = predictor.infer_lesion(args.image, args.seg)
            print("\n Prediction Result:")
            print("-" * 30)
            print(f" Class:      {result['class_name']} (ID: {result['class_id']})")
            print(f" Confidence: {result['confidence']:.2%}")
            print(f" Volume:     {result['volume_voxels']} voxels")
            print("-" * 30)
        except ValueError as e:
            print(f"Error: {e}")


# ================= Configuration =================
def config_train(parser: argparse.ArgumentParser) -> None:
    """Configures an existing parser for 'train' and binds the logic."""
    parser.add_argument("--data", required=True, help="Path to training features (Parquet)")
    parser.add_argument("--output-dir", required=True, help="Folder to save model.pkl")
    parser.add_argument("--extractor-config", required=True, help="JSON config from features")
    parser.add_argument("--model", choices=["logistic", "tree", "xgboost"], default="logistic")
    parser.add_argument("--class-config", help="JSON map of class names (e.g. {'0': 'Tumor'})")
    parser.add_argument("--depth", type=int, default=5, help="Tree depth")
    parser.set_defaults(func=run_train)


def config_eval(parser: argparse.ArgumentParser) -> None:
    """Configures an existing parser for 'eval' and binds the logic."""
    parser.add_argument("--data", required=True, help="Path to test features (Parquet)")
    parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    parser.add_argument("--output-dir", required=True, help="Folder to save results")
    parser.set_defaults(func=run_eval)


def config_infer(parser: argparse.ArgumentParser) -> None:
    """Configures an existing parser for 'infer' and binds the logic."""
    parser.add_argument("--image", required=True, help="Path to CT image")
    parser.add_argument("--seg", required=True, help="Path to segmentation mask")
    parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    parser.add_argument("--output", help="Path to save results CSV (optional)")
    parser.set_defaults(func=run_infer)


def add_subparsers(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser("train", help="Train a model on extracted features")
    config_train(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    config_eval(eval_parser)

    infer_parser = subparsers.add_parser("infer", help="Run inference on a new image")
    config_infer(infer_parser)


# ================= Standalone CLI =================
def main() -> None:
    parser = get_base_parser("Classifier Module")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")
    add_subparsers(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
