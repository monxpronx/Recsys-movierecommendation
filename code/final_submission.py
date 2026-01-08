import pandas as pd
import numpy as np

# ===============================
# 설정
# ===============================
SUB_PATH = "../data/eval/sample_submission.csv"

ADMM_PATH = "output/admm_top100.csv"
LGCN_PATH = "output/lightgcn_top100.csv"
GRU_PATH  = "output/gru4rec_top100.csv"

OUT_PATH  = "final_submission.csv"

# 모델 가중치 (Recall@10 기준)
WEIGHTS = {
    "admm": 0.7,
    "lightgcn": 0.2,
    "gru4rec": 0.1
}

# ===============================
# 유틸 함수
# ===============================
def normalize_score(df, score_col="score"):
    min_s = df[score_col].min()
    max_s = df[score_col].max()
    if max_s > min_s:
        df["score_norm"] = (df[score_col] - min_s) / (max_s - min_s)
    else:
        df["score_norm"] = 0.0
    return df

# ===============================
# 1) ADMM
# ===============================
admm = pd.read_csv(ADMM_PATH)   # user,item,rank,score
admm = normalize_score(admm)
admm["model"] = "admm"

# ===============================
# 2) LightGCN
# ===============================
lgcn = pd.read_csv(LGCN_PATH)   # user,item,score
lgcn = normalize_score(lgcn)
lgcn["model"] = "lightgcn"

# ===============================
# 3) GRU4Rec
# ===============================
gru = pd.read_csv(GRU_PATH)     # user,item,score
gru = normalize_score(gru)
gru["model"] = "gru4rec"

# ===============================
# 4) 앙상블 점수 계산 (Soft Voting)
# ===============================
all_df = pd.concat([admm, lgcn, gru], ignore_index=True)

def ensemble_score(row):
    return WEIGHTS[row["model"]] * row["score_norm"]

all_df["ensemble_score"] = all_df.apply(ensemble_score, axis=1)

# user-item 단위로 합산
final_scores = (
    all_df
    .groupby(["user", "item"], as_index=False)["ensemble_score"]
    .sum()
)

# ===============================
# 5) 제출 대상 user 기준 Top-10 생성
# ===============================
sub = pd.read_csv(SUB_PATH)   # user,item (item은 더미)
target_users = sub["user"].unique()

rows = []

for uid in target_users:
    cand = final_scores[final_scores["user"] == uid]

    if len(cand) == 0:
        continue  # 필요하면 popularity로 대체 가능

    top10 = (
        cand.sort_values("ensemble_score", ascending=False)
            .head(10)
    )

    for it in top10["item"].tolist():
        rows.append([uid, it])

# ===============================
# 6) 최종 제출 파일 저장
# ===============================
submit = pd.DataFrame(rows, columns=["user", "item"])
submit.to_csv(OUT_PATH, index=False)

print(f"Saved: {OUT_PATH}")
print("총 행 수:", len(submit))
