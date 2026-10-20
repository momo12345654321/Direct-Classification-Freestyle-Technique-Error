import joblib
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier


TRAINING_SET = (
    "data/swimming/random_forest/frame_features.csv"
)

OUTPUT_MODEL = (
    "work_dirs/random_forest/technique_errors_rf.joblib"
)

os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)

FEATURES = [
    f"feature_{i}"
    for i in range(51)
]


df = pd.read_csv(TRAINING_SET, low_memory=False)
df.fillna(0, inplace=True)
df = df[df["split"] == "train"].reset_index(drop=True)

clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    n_jobs=-1,
    random_state=2017,
)

clf.fit(
    df[FEATURES],
    df["ManualLabel"],
)

joblib.dump(clf, OUTPUT_MODEL)

print("Training frames:", len(df))
print("Features per frame:", clf.n_features_in_)
print("Classes learned:", clf.classes_)
print("Trees:", len(clf.estimators_))
print("Saved:", OUTPUT_MODEL)
