import argparse
import csv
import os
import pickle
from collections import Counter

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a PYSKL swimming-cycle dataset."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--pose-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--val-swimmers",
        nargs="*",
        default=[],
        help="Swimmer IDs assigned only to validation",
    )
    parser.add_argument(
        "--test-swimmers",
        nargs="*",
        default=[],
        help="Swimmer IDs assigned only to final testing",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()

    annotations = []
    train_ids = []
    val_ids = []
    test_ids = []
    label_counts = Counter()
    split_label_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    used_ids = set()

    val_swimmers = set(args.val_swimmers)
    test_swimmers = set(args.test_swimmers)

    overlap = val_swimmers & test_swimmers
    if overlap:
        raise ValueError(
            f"Swimmers cannot be in both val and test: {sorted(overlap)}"
        )

    skipped_pending = 0
    skipped_unusable = 0
    skipped_unlabeled = 0

    with open(args.csv, newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)

        error_columns = [
            "dropped_elbow",
            "straight_arm_pull",
            "shortened_pull",
        ]
        missing_columns = [
            column
            for column in error_columns
            if column not in rows.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                f"Missing error columns: {missing_columns}. "
                f"Available columns: {rows.fieldnames}"
            )

        for row_number, row in enumerate(rows, start=2):
            swimmer = row["swimmer"].strip()

            if swimmer in val_swimmers:
                split_name = "val"
            elif swimmer in test_swimmers:
                split_name = "test"
            else:
                split_name = "train"

            label_status = row.get("label_status", "ready").strip().lower()

            if label_status and label_status != "ready":
                skipped_pending += 1
                continue

            if row["usable"].strip() != "1":
                skipped_unusable += 1
                continue

            label_texts = [
                row[column].strip()
                for column in error_columns
            ]

            if any(value not in {"0", "1"} for value in label_texts):
                skipped_unlabeled += 1
                continue

            error_values = [int(value) for value in label_texts]

            if sum(error_values) > 1:
                raise ValueError(
                    f"CSV row {row_number} has multiple errors marked: "
                    f"{dict(zip(error_columns, error_values))}"
                )

            if error_values == [0, 0, 0]:
                label = 0
            elif error_values == [1, 0, 0]:
                label = 1
            elif error_values == [0, 1, 0]:
                label = 2
            elif error_values == [0, 0, 1]:
                label = 3
            else:
                raise ValueError(
                    f"Unsupported labels at CSV row {row_number}: "
                    f"{error_values}"
                )

            cycle_number = int(row["cycle_num"])
            cycle_id = f"c{cycle_number:04d}"

            if cycle_id in used_ids:
                raise ValueError(f"Duplicate cycle ID {cycle_id} at CSV row {row_number}")

            used_ids.add(cycle_id)

            video_id = row["video"].strip()
            pose_path = os.path.join(
                args.pose_dir,
                f"{video_id}.pkl",
            )

            if not os.path.isfile(pose_path):
                raise FileNotFoundError(
                    f"CSV row {row_number} requires missing file: "
                    f"{pose_path}"
                )

            start_time = float(row["start"])
            end_time = float(row["end"])

            if end_time <= start_time:
                raise ValueError(
                    f"Invalid time interval at CSV row {row_number}: "
                    f"{start_time} to {end_time}"
                )


            with open(pose_path, "rb") as pose_file:
                full_anno = pickle.load(pose_file)

            full_keypoint = full_anno["keypoint"]
            full_score = full_anno["keypoint_score"]
            total_video_frames = full_keypoint.shape[1]

            start_frame = int(round(start_time * args.fps))
            end_frame = int(round(end_time * args.fps))

            start_frame = max(0, start_frame)
            end_frame = min(total_video_frames, end_frame)

            if end_frame <= start_frame:
                raise ValueError(
                    f"Cycle {cycle_id} has no frames after clipping: "
                    f"{start_frame}:{end_frame}"
                )


            cycle_keypoint = full_keypoint[
                :1,
                start_frame:end_frame,
                :,
                :
            ].copy()

            cycle_score = full_score[
                :1,
                start_frame:end_frame,
                :
            ].copy()

            if cycle_keypoint.shape[1] < 10:
                raise ValueError(
                    f"Cycle {cycle_id} has only "
                    f"{cycle_keypoint.shape[1]} frames"
                )

            if cycle_keypoint.shape[2:] != (17, 2):
                raise ValueError(
                    f"Unexpected keypoint shape for {cycle_id}: "
                    f"{cycle_keypoint.shape}"
                )

            if cycle_score.shape != cycle_keypoint.shape[:-1]:
                raise ValueError(
                    f"Score/keypoint mismatch for {cycle_id}: "
                    f"{cycle_score.shape} versus "
                    f"{cycle_keypoint.shape}"
                )

            if not np.isfinite(cycle_keypoint).all():
                raise ValueError(
                    f"Non-finite keypoints in {cycle_id}"
                )

            if not np.isfinite(cycle_score).all():
                raise ValueError(
                    f"Non-finite scores in {cycle_id}"
                )

            annotation = {
                "frame_dir": cycle_id,
                "label": label,
                "img_shape": tuple(full_anno["img_shape"]),
                "original_shape": tuple(
                    full_anno["original_shape"]
                ),
                "total_frames": cycle_keypoint.shape[1],
                "keypoint": cycle_keypoint,
                "keypoint_score": cycle_score,
                "source_video": video_id,
                "swimmer_id": swimmer,
                "session_id": row["session"].strip(),
                "view": row["side"].strip(),
                "start_time": start_time,
                "end_time": end_time,
            }

            annotations.append(annotation)
            if split_name == "train":
                train_ids.append(cycle_id)
            elif split_name == "val":
                val_ids.append(cycle_id)
            else:
                test_ids.append(cycle_id)
            label_counts[label] += 1
            split_label_counts[split_name][label] += 1

# ---- Balance the training split by oversampling minority classes ----
    train_id_set = set(train_ids)
    class_to_train_annos = {0: [], 1: [], 2: [], 3: []}
    for anno in annotations:
        if anno["frame_dir"] in train_id_set:
            class_to_train_annos[anno["label"]].append(anno)

    max_count = max(len(v) for v in class_to_train_annos.values())
    dup_counter = 0

    for label, annos_for_class in class_to_train_annos.items():
        current_count = len(annos_for_class)
        if current_count == 0:
            continue
        needed = max_count - current_count
        i = 0
        while needed > 0:
            source = annos_for_class[i % current_count]
            dup_counter += 1
            new_id = f"{source['frame_dir']}_dup{dup_counter:04d}"
            new_anno = dict(source)
            new_anno["frame_dir"] = new_id
            annotations.append(new_anno)
            train_ids.append(new_id)
            split_label_counts["train"][label] += 1
            label_counts[label] += 1
            i += 1
            needed -= 1

    print(f"Balancing: added {dup_counter} duplicated training cycles "
          f"(train only; val/test untouched)")
    
    if not annotations:
        raise RuntimeError("No eligible annotations were found.")

    dataset = {
        "split": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "annotations": annotations,
    }


    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "wb") as output_file:
        pickle.dump(dataset, output_file)

    print(f"Saved: {args.output}")
    print(f"Training cycles: {len(train_ids)}")
    print(f"Validation cycles: {len(val_ids)}")
    print(f"Testing cycles: {len(test_ids)}")
    print(f"Label 0 (correct): {label_counts[0]}")
    print(f"Label 1 (dropped elbow): {label_counts[1]}")
    print(f"Label 2 (straight-arm pull): {label_counts[2]}")
    print(f"Label 3 (shortened pull): {label_counts[3]}")
    for split_name, ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ]:
        counts = split_label_counts[split_name]
        print(f"{split_name}: {len(ids)} cycles; labels "
              f"0={counts[0]}, 1={counts[1]}, "
              f"2={counts[2]}, 3={counts[3]}")
    print(f"Skipped pending: {skipped_pending}")
    print(f"Skipped unusable: {skipped_unusable}")
    print(f"Skipped unlabeled: {skipped_unlabeled}")


if __name__ == "__main__":
    main()
