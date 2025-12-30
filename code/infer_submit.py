import pandas as pd
import numpy as np
import torch
from gru4rec_pytorch import GRU4Rec

TRAIN_PATH = "../data/train/train_ratings.csv"
SUB_PATH   = "../data/eval/sample_submission.csv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===== 1) 모델 로드 =====
gru = GRU4Rec.loadmodel("gru4rec_model.pt", device=DEVICE)
gru.model.eval()
itemidmap = gru.data_iterator.itemidmap  # index=원본 ItemId, value=ItemIdx

# ===== 2) submission: 추천 대상 user 목록만 =====
sub = pd.read_csv(SUB_PATH)
col_user = "user"
users = sub[col_user].unique()

# ===== 3) 로그: 유저 시퀀스는 train에서 구성 =====
log = pd.read_csv(TRAIN_PATH)

col_item = "item"
col_time = "time"   # train에는 time이 있다고 가정

log = log[[col_user, col_item, col_time]].copy()
log.dropna(subset=[col_user, col_item, col_time], inplace=True)
log[col_time] = log[col_time].astype("int64")
log.sort_values([col_user, col_time], inplace=True)

# 학습에 없던 아이템 제거 (추천 엔진이 모르는 item은 입력으로 못 씀)
log = log[log[col_item].isin(itemidmap.index)]

# 유저별 최근 N개로 시퀀스 구성 (제출용)
N_RECENT = 50
user2seq = (
    log.groupby(col_user)[col_item]
       .apply(lambda s: s.tail(N_RECENT).tolist())
       .to_dict()
)

def recommend_from_sequence(seq_item_ids, k=10):
    if seq_item_ids is None or len(seq_item_ids) < 2:
        return None

    idxs = itemidmap.loc[seq_item_ids].values.astype(np.int64)

    H = [torch.zeros((1, h), dtype=torch.float32, device=DEVICE) for h in gru.layers]

    with torch.no_grad():
        scores = None
        for item_idx in idxs:
            x = torch.tensor([item_idx], dtype=torch.long, device=DEVICE)
            scores = gru.model.forward(x, H, Y=None, training=False)

    scores = scores[0]
    seen = torch.tensor(np.unique(idxs), device=DEVICE)
    scores[seen] = -1e9

    _, topk_idx = torch.topk(scores, k)
    rec_item_ids = itemidmap.index[topk_idx.cpu().numpy()].tolist()
    return rec_item_ids

out_rows = []

for uid in users:
    seq = user2seq.get(uid, None)
    rec = recommend_from_sequence(seq, k=10)
    if rec is None:
        continue  # 이 유저는 다른 모델로 채울 예정

    # 유저당 10개 → 10행으로 풀기
    for it in rec:
        out_rows.append([uid, it])

out = pd.DataFrame(out_rows, columns=["user", "item"])
out.to_csv("gru4rec_only.csv", index=False)
print("Saved: gru4rec_only.csv rows:", len(out))