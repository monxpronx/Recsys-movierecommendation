# 앙상블: ADMMSLIM(0.4) / EASE(0.2) / CDAE(0.4)


import pandas as pd
import numpy as np

# CSV 로드 (rank-only submission)
admm = pd.read_csv("ADMMSLIM.csv").rename(
    columns={"user": "user_id", "item": "item_id"}
)
ease = pd.read_csv("EASE.csv").rename(
    columns={"user": "user_id", "item": "item_id"}
)
cdae = pd.read_csv("CDAE.csv").rename(
    columns={"user": "user_id", "item": "item_id"}
)

# user별 rank 생성 (순서 = rank)
admm["rank_admm"] = admm.groupby("user_id").cumcount() + 1
ease["rank_ease"] = ease.groupby("user_id").cumcount() + 1
cdae["rank_cdae"] = cdae.groupby("user_id").cumcount() + 1

# outer merge
df = admm.merge(ease, on=["user_id", "item_id"], how="outer")
df = df.merge(cdae, on=["user_id", "item_id"], how="outer")

# 결측 rank 처리
MAX_RANK = 1000
df[["rank_admm", "rank_ease", "rank_cdae"]] = (
    df[["rank_admm", "rank_ease", "rank_cdae"]].fillna(MAX_RANK)
)

# 가중 rank 앙상블
df["final_rank"] = (
    0.4 * df["rank_admm"]
    + 0.2 * df["rank_ease"]
    + 0.4 * df["rank_cdae"]
)

# user별 Top-K
TOP_K = 10
df_final = (
    df
    .sort_values(["user_id", "final_rank"], ascending=[True, True])
    .groupby("user_id", sort=False)
    .head(TOP_K)
)

# CSV 저장
(
    df_final[["user_id", "item_id"]]
    .rename(columns={"user_id": "user", "item_id": "item"})
    .to_csv("ensemble_outputs/ensemble_02.csv", index=False)
)