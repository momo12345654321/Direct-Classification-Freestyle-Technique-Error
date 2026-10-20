import argparse
import pickle

import numpy as np
from mmcv import Config
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from pyskl.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('results_pkl')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    with open(args.results_pkl, 'rb') as f:
        outputs = pickle.load(f)

    if len(outputs) != len(dataset.video_infos):
        raise ValueError(
            f"Mismatch: {len(outputs)} predictions vs "
            f"{len(dataset.video_infos)} test videos"
        )

    y_true = []
    y_pred = []

    for item, scores in zip(dataset.video_infos, outputs):
        if 'label' not in item:
            raise KeyError(f"No 'label' field found in video_infos entry: {item}")
        y_true.append(int(item['label']))
        y_pred.append(int(np.argmax(scores)))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_names = ['correct', 'dropped_elbow', 'straight_arm_pull', 'shortened_pull']
    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    overall_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average='macro', zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    print(f"Test videos: {len(y_true)}")
    print(f"Classes actually present in y_true/y_pred: {labels_present}")
    print()
    print(f"Overall accuracy:   {overall_acc:.4f}")
    print(f"Balanced accuracy:  {balanced_acc:.4f}")
    print(f"Macro precision:    {macro_p:.4f}")
    print(f"Macro recall:       {macro_r:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print()
    print("Per-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name:20s} precision={per_p[i]:.4f}  recall={per_r[i]:.4f}  "
              f"f1={per_f1[i]:.4f}  support={per_support[i]}")
    print()
    print("Confusion matrix (rows=true, cols=predicted), order = "
          "[correct, dropped_elbow, straight_arm_pull, shortened_pull]:")
    print(cm)


if __name__ == '__main__':
    main()
