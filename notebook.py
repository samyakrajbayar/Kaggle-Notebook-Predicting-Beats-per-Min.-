import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# ==============================
# Load Data
# ==============================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

id_col = "id"
target_col = "BeatsPerMinute"

X = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test = test.drop(columns=[id_col])

# ==============================
# Target transform
# ==============================
y_log = np.log1p(y)

# ==============================
# CV setup
# ==============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y_log)):
    print(f"FOLD {fold+1}")
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[trn_idx], y_log.iloc[val_idx]

    model = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=128,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)

    val_pred = np.expm1(model.predict(X_val))
    test_pred = np.expm1(model.predict(X_test))

    oof[val_idx] = val_pred
    preds += test_pred / kf.n_splits

# ==============================
# CV Score
# ==============================
rmse = mean_squared_error(y, oof, squared=False)
print(f"CV RMSE: {rmse:.4f}")

# ==============================
# Submission
# ==============================
sub = sample_sub.copy()
sub[target_col] = preds
sub.to_csv("submission.csv", index=False)
print("submission.csv written!")
FOLD 1
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.021837 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2295
[LightGBM] [Info] Number of data points in the train set: 419331, number of used features: 9
[LightGBM] [Info] Start training from score 4.762035
FOLD 2
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.019478 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2295
[LightGBM] [Info] Number of data points in the train set: 419331, number of used features: 9
[LightGBM] [Info] Start training from score 4.761895
FOLD 3
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.019782 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2295
[LightGBM] [Info] Number of data points in the train set: 419331, number of used features: 9
[LightGBM] [Info] Start training from score 4.761866
FOLD 4
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.023416 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2295
[LightGBM] [Info] Number of data points in the train set: 419331, number of used features: 9
[LightGBM] [Info] Start training from score 4.761755
FOLD 5
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.019572 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 2295
[LightGBM] [Info] Number of data points in the train set: 419332, number of used features: 9
[LightGBM] [Info] Start training from score 4.761709
CV RMSE: 26.7037
submission.csv written!
