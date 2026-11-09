import argparse
import json
from pathlib import Path

from fraud_detection.data.core import read_csv, synthetic
from fraud_detection.evaluation.reporting import eda
from fraud_detection.modeling.train import run


def main():
    parser = argparse.ArgumentParser(description="Historical fraud review prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--output", type=Path, default=Path("data/synthetic/fixture.csv"))
    audit = sub.add_parser("eda")
    audit.add_argument("--csv", required=True)
    audit.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--csv", required=True)
    train.add_argument("--kind", choices=["real", "synthetic"], required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--artifacts", required=True)
    train.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    if args.command == "fixture":
        synthetic(args.output)
        print(f"SYNTHETIC fixture written to {args.output}")
    elif args.command == "eda":
        eda(read_csv(args.csv), args.output)
    else:
        run(
            args.csv,
            args.output,
            args.artifacts,
            json.loads(Path(args.config).read_text()),
            args.kind,
        )


if __name__ == "__main__":
    main()
