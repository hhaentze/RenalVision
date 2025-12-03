"""
Main command-line interface for the Cyst Classifier.
"""

import argparse

from classifier import eval, inference, train


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Renal Vision: Classifier Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # ================= TRAIN =================
    train_parser = subparsers.add_parser("train", help="Train a model on extracted features")
    train_parser.add_argument("--data", required=True, help="Path to training features (Parquet)")
    train_parser.add_argument("--output-dir", required=True, help="Folder to save model.pkl")
    train_parser.add_argument(
        "--model", choices=["logistic", "tree", "xgboost"], default="logistic"
    )
    train_parser.add_argument(
        "--class-config", help="JSON map of class names (e.g. {'0': 'Tumor'})"
    )
    train_parser.add_argument(
        "--extractor-config", help="JSON config from feature_handler (optional)"
    )
    train_parser.add_argument("--depth", type=int, default=5, help="Tree depth")

    # ================= EVAL =================
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--data", required=True, help="Path to test features (Parquet)")
    eval_parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    eval_parser.add_argument("--output-dir", required=True, help="Folder to save results")

    # ================= INFER =================
    infer_parser = subparsers.add_parser("infer", help="Run inference on a new image")
    infer_parser.add_argument("--image", required=True, help="Path to CT image")
    infer_parser.add_argument("--seg", required=True, help="Path to segmentation mask")
    infer_parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    infer_parser.add_argument("--output", help="Path to save results CSV (optional)")

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "train":
        train.run_training(
            data_path=args.data,
            output_dir=args.output_dir,
            model_type=args.model,
            class_config_path=args.class_config,
            extractor_config_path=args.extractor_config,
            tree_max_depth=args.depth,
        )

    elif args.command == "eval":
        eval.run_evaluation(
            data_path=args.data,
            model_path=args.model,
            output_dir=args.output_dir,
        )

    elif args.command == "infer":
        predictor = inference.LesionPredictor(model_path=args.model)

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


if __name__ == "__main__":
    main()
