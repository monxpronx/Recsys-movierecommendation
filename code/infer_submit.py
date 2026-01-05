import pandas as pd
import numpy as np
import torch
from gru4rec_pytorch import GRU4Rec

# ===============================
# 설정
# ===============================
TRAIN_PATH = "../data/train/train_ratings.csv"
SUB_PATH   = "../data/eval/sample_submission.csv"
MODEL_PATH = "gru4rec_model.pt"
OUT_PATH   = "gru4rec_top100.csv"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

col_user = "user"
col_item = "item"
col_time = "time"

N_RECENT = 50
TOPK = 100

# ===============================
# 1) 모델 로드
# ===============================
gru = GRU4Rec.loadmodel(MODEL_PATH, device=DEVICE)
gru.model.eval()

itemidmap = gru.data_iterator.itemidmap  # index=원본 ItemId, value=ItemIdx
layers = gru.layers

print("Model loaded. #items:", len(itemidmap))

# ===============================
# 2) submission 대상 user
# ===============================
sub = pd.read_csv(SUB_PATH)
users = sub[col_user].unique()
print("Target users:", len(users))

# ===============================
# 3) train 로그로 user 시퀀스 구성
# ===============================
log = pd.read_csv(TRAIN_PATH)
log = log[[col_user, col_item, col_time]].copy()
log.dropna(subset=[col_user, col_item, col_time], inplace=True)
log[col_time] = log[col_time].astype("int64")
log.sort_values([col_user, col_time], inplace=True)

# 학습에 없던 아이템 제거
log = log[log[col_item].isin(itemidmap.index)]

user2seq = (
    log.groupby(col_user)[col_item]
       .apply(lambda s: s.tail(N_RECENT).tolist())
       .to_dict()
)

print("Users with history:", len(user2seq))

# ===============================
# 4) 추천 함수
# ===============================
def recommend_from_sequence(seq_item_ids, k=100):
    if seq_item_ids is None or len(seq_item_ids) < 2:
        return None

    idxs = itemidmap.loc[seq_item_ids].values.astype(np.int64)

    # hidden state 초기화
    H = [
        torch.zeros((1, h), dtype=torch.float32, device=DEVICE)
        for h in layers
    ]

    with torch.no_grad():
        scores = None
        for item_idx in idxs:
            x = torch.tensor([item_idx], dtype=torch.long, device=DEVICE)
            scores = gru.model.forward(x, H, Y=None, training=False)

    scores = scores[0]  # (num_items,)

    # 이미 본 아이템 제거
    seen = torch.tensor(np.unique(idxs), device=DEVICE)
    scores[seen] = -1e9

    topk_scores, topk_idx = torch.topk(scores, k)

    rec_item_ids = itemidmap.index[topk_idx.cpu().numpy()].tolist()
    rec_scores = topk_scores.cpu().numpy().tolist()

    results = []
    for rank, (it, sc) in enumerate(zip(rec_item_ids, rec_scores), start=1):
        results.append((it, rank, float(sc)))

    return results

# ===============================
# 5) 전체 user 추론
# ===============================
out_rows = []

for uid in users:
    seq = user2seq.get(uid, None)
    rec = recommend_from_sequence(seq, k=TOPK)

    if rec is None:
        continue  # cold user → 앙상블에서 다른 모델이 채움

    # ✅ 반드시 이 안에 있어야 함
    for it, rank, score in rec:
        out_rows.append([uid, it, rank, score])

# ===============================
# 6) 저장
# ===============================
out = pd.DataFrame(
    out_rows,
    columns=["user", "item", "rank", "score"]
)

out.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Total rows:", len(out))
print("Unique users:", out["user"].nunique())
