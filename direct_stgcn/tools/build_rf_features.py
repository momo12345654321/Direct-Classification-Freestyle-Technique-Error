import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pyskl.datasets.pipelines.pose_related import PreNormalize2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-swimmers", nargs="*", default=[])
    parser.add_argument("--test-swimmers", nargs="*", default=[])
    return parser.parse_args()


args = parse_args()
val_swimmers = set(str(s) for s in args.val_swimmers)
test_swimmers = set(str(s) for s in args.test_swimmers)

overlap = val_swimmers & test_swimmers
if overlap:
    raise ValueError(f"Swimmers cannot be in both val and test: {sorted(overlap)}")

CYCLES_PATH = "data/swimming/metadata/cycles.csv"
POSE_DIR = Path("data/swimming/full_video_poses")
OUTPUT_PATH = Path(
    "data/swimming/random_forest/frame_features.csv"
)

FPS = 30

ERROR_COLUMNS = [
    "dropped_elbow",
    "straight_arm_pull",
    "shortened_pull",
]


cycles = pd.read_csv(CYCLES_PATH)

# Only use completed, usable annotations.
cycles = cycles[
    (cycles["usable"] == 1)
    & (
        cycles["label_status"]
        .astype(str)
        .str.lower()
        == "ready"
    )
].copy()

cycles[ERROR_COLUMNS] = (
    cycles[ERROR_COLUMNS]
    .fillna(0)
    .astype(int)
)

# Every cycle must have no more than one error.
if (cycles[ERROR_COLUMNS].sum(axis=1) > 1).any():
    raise ValueError("A cycle has more than one error label.")

# Convert the three error columns into one class.
cycles["ManualLabel"] = 0
cycles.loc[
    cycles["dropped_elbow"] == 1,
    "ManualLabel",
] = 1
cycles.loc[
    cycles["straight_arm_pull"] == 1,
    "ManualLabel",
] = 2
cycles.loc[
    cycles["shortened_pull"] == 1,
    "ManualLabel",
] = 3

rows = []
pose_cache = {}

for cycle in cycles.itertuples(index=False):
    pose_path = POSE_DIR / f"{cycle.video}.pkl"

    if not pose_path.exists():
        raise FileNotFoundError(pose_path)

    if cycle.video not in pose_cache:
        with open(pose_path, "rb") as file:
            pose_cache[cycle.video] = pickle.load(file)

    pose = pose_cache[cycle.video]

    start_frame = round(cycle.start * FPS)
    end_frame = round(cycle.end * FPS)

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, pose["total_frames"])

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid frame range for {cycle.video}, "
            f"cycle {cycle.cycle_num}"
        )

    sample = {
        "keypoint": pose["keypoint"][
            :, start_frame:end_frame
        ].copy(),
        "keypoint_score": pose["keypoint_score"][
            :, start_frame:end_frame
        ].copy(),
        "img_shape": pose["img_shape"],
        "total_frames": end_frame - start_frame,
    }

    # Same default fix normalization as ST-GCN.
    sample = PreNormalize2D()(sample)

    # [1, T, 17, 3] -> [T, 51]
    features = sample["keypoint"][0].reshape(-1, 51)

    for local_frame, feature_vector in enumerate(features):
        row = {
            f"feature_{i}": value
            for i, value in enumerate(feature_vector)
        }

        swimmer_str = str(cycle.swimmer).strip()
        if swimmer_str in val_swimmers:
            split_name = "val"
        elif swimmer_str in test_swimmers:
            split_name = "test"
        else:
            split_name = "train"

        row["swimmer"] = cycle.swimmer
        row["video"] = cycle.video
        row["cycle_num"] = cycle.cycle_num
        row["frame"] = start_frame + local_frame
        row["ManualLabel"] = cycle.ManualLabel
        row["split"] = split_name
        rows.append(row)

output = pd.DataFrame(rows)

train_mask = output["split"] == "train"
train_df = output[train_mask]
other_df = output[~train_mask]

cycle_key_cols = ["swimmer", "video", "cycle_num"]
cycle_labels = train_df[cycle_key_cols + ["ManualLabel"]].drop_duplicates(subset=cycle_key_cols)

class_to_cycles = {
    label: cycle_labels[cycle_labels["ManualLabel"] == label][cycle_key_cols].values.tolist()
    for label in [0, 1, 2, 3]
}


# Exact frame-level balancing: duplicate whole cycles until adding the next
# whole cycle would overshoot the target, then top off the remainder with a
# random subsample of individual frames from one more cycle -- so every
# class ends at EXACTLY max_frame_count frames, not just approximately.
frame_counts_by_class = train_df.groupby("ManualLabel").size().to_dict()
max_frame_count = max(frame_counts_by_class.values())

balanced_frames = [train_df]
dup_counter = 0

for label, cycles in class_to_cycles.items():
    current_count = len(cycles)
    if current_count == 0:
        continue
    current_frame_count = frame_counts_by_class.get(label, 0)
    frames_needed = max_frame_count - current_frame_count
    i = 0
    while frames_needed > 0:
        swimmer, video, cycle_num = cycles[i % current_count]
        cycle_frames = train_df[
            (train_df["swimmer"] == swimmer)
            & (train_df["video"] == video)
            & (train_df["cycle_num"] == cycle_num)
        ].copy()
        n_cycle_frames = len(cycle_frames)
        dup_counter += 1

        if n_cycle_frames <= frames_needed:
            cycle_frames["cycle_num"] = f"{cycle_num}_dup{dup_counter}"
            balanced_frames.append(cycle_frames)
            frames_needed -= n_cycle_frames
        else:
            sampled = cycle_frames.sample(
                n=frames_needed, random_state=2017, replace=False
            ).copy()
            sampled["cycle_num"] = f"{cycle_num}_dup{dup_counter}"
            balanced_frames.append(sampled)
            frames_needed = 0

        i += 1

train_df_balanced = pd.concat(balanced_frames, ignore_index=True)
output = pd.concat([train_df_balanced, other_df], ignore_index=True)

print(f"Balancing: added {dup_counter} duplicated cycles/partial-cycles "
      f"(exact frame-level target: {max_frame_count} frames/class; train only; val/test untouched)")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print("Training swimmers:", sorted(output[output["split"] == "train"]["swimmer"].unique()))
print("Output shape:", output.shape)
print("Feature columns: 51")
print("Frame labels:")
print(output["ManualLabel"].value_counts().sort_index())
print("Saved:", OUTPUT_PATH) 
