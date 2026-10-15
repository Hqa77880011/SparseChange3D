import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", nargs="+", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    records = [(Path(path).parent.name, json.loads(Path(path).read_text())) for path in args.runs]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for metric in ["signed_macro_f1", "added_iou", "removed_iou", "stable_false_change_percent", "unknown_fraction", "minimum_detectable_size"]:
        available = [(label, row[metric]) for label, row in records if row.get(metric) is not None]
        if available:
            labels, values = zip(*available); figure, axis = plt.subplots(); axis.bar(labels, values); axis.set_ylabel(metric); axis.tick_params(axis="x", rotation=30); figure.savefig(output / f"{metric}.png", dpi=180, bbox_inches="tight"); plt.close(figure)


if __name__ == "__main__":
    main()
