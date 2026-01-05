import pandas as pd
import numpy as np
import torch

from gru4rec_pytorch import GRU4Rec  # 파일명 맞추세요: gru4rec_pytorch.py

DATA_PATH = "../data/train/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===== 1) 로드 =====
df = pd.read_csv(DATA_PATH + "train_ratings.csv")

# 실제 컬럼명에 맞게 바꾸세요
col_user = "user"
col_item = "item"
col_time = "time"

df = df[[col_user, col_item, col_time]].copy()
df.dropna(subset=[col_user, col_item, col_time], inplace=True)
df[col_time] = df[col_time].astype("int64")
df.sort_values([col_user, col_time], inplace=True)

# ===== 2) 세션화(1시간 gap) =====
SESSION_GAP = 60 * 60  # 1 hour
df["prev_time"] = df.groupby(col_user)[col_time].shift(1)
df["gap"] = df[col_time] - df["prev_time"]
df["new_sess"] = df["prev_time"].isna() | (df["gap"] >= SESSION_GAP)
df["sess_no"] = df.groupby(col_user)["new_sess"].cumsum().astype("int64")

df["SessionId"] = df[col_user].astype(str) + "_" + df["sess_no"].astype(str)
df.rename(columns={col_item: "ItemId", col_time: "Time"}, inplace=True)
train_sessions = df[["SessionId", "ItemId", "Time"]].copy()

# 학습용: 세션 길이 2 미만 제거
sess_len = train_sessions.groupby("SessionId").size()
train_sessions = train_sessions[train_sessions["SessionId"].isin(sess_len[sess_len >= 2].index)]
train_sessions.sort_values(["SessionId", "Time"], inplace=True)

print("Train sessions:", train_sessions["SessionId"].nunique(), "rows:", len(train_sessions))

# ===== 3) 학습 =====
gru = GRU4Rec(
    layers=[200],
    loss="cross-entropy",
    batch_size=64,
    n_epochs=10,
    learning_rate=0.02,
    momentum=0.0,
    n_sample=1024,
    sample_alpha=0.5,
    dropout_p_embed=0.0,
    dropout_p_hidden=0.2,
    constrained_embedding=True,
    device=torch.device(DEVICE),
)

gru.fit(train_sessions, item_key="ItemId", session_key="SessionId", time_key="Time")

# ===== 4) 저장 =====
gru.savemodel("gru4rec_model.pt")
print("Saved:", "gru4rec_model.pt")
