import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

FEATURES = [f"feature_{i}" for i in range(51)]
CLASS_NAMES = ['correct', 'dropped_elbow', 'straight_arm_pull', 'shortened_pull']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/swimming/random_forest/frame_features.csv")
    parser.add_argument("--model", default="work_dirs/random_forest/technique_errors_rf.joblib")
    parser.add_argument("--split", required=True, choices=["val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    df = df[df["split"] == args.split].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No rows found for split='{args.split}'")

    df.fillna(0, inplace=True)

    clf = joblib.load(args.model)

    frame_preds = clf.predict(df[FEATURES])
    df["frame_pred"] = frame_preds

    frame_acc = accuracy_score(df["ManualLabel"], df["frame_pred"])

    cycle_key_cols = ["swimmer", "video", "cycle_num"]
    grouped = df.groupby(cycle_key_cols)

    y_true = []
    y_pred = []
    for _, group in grouped:
        true_label = group["ManualLabel"].iloc[0]
        if not (group["ManualLabel"] == true_label).all():
            raise ValueError(f"Inconsistent ManualLabel within one cycle: {group[cycle_key_cols].iloc[0].to_dict()}")
        pred_label = group["frame_pred"].mode().iloc[0]
        y_true.append(int(true_label))
        y_pred.append(int(pred_label))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average='macro', zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    print(f"Split: {args.split}")
    print(f"Frame-level rows: {len(df)}")
    print(f"Frame-level accuracy: {frame_acc:.4f}")
    print(f"Cycles (after majority-vote aggregation): {len(y_true)}")
    print()
    print(f"Overall accuracy:   {overall_acc:.4f}")
    print(f"Balanced accuracy:  {balanced_acc:.4f}")
    print(f"Macro precision:    {macro_p:.4f}")
    print(f"Macro recall:       {macro_r:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print()
    print("Per-class metrics:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name:20s} precision={per_p[i]:.4f}  recall={per_r[i]:.4f}  "
              f"f1={per_f1[i]:.4f}  support={per_support[i]}")
    print()
    print("Confusion matrix (rows=true, cols=predicted), order = "
          "[correct, dropped_elbow, straight_arm_pull, shortened_pull]:")
    print(cm)


if __name__ == '__main__':
    main()
