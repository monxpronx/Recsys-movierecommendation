# 앙상블: ADMMSLIM(0.45) / EASE(0.30) / CDAE(0.25)


import pandas as pd
import numpy as np

admm = pd.read_csv("ADMMSLIM.csv").rename(
    columns={"user": "user_id", "item": "item_id"}
)
ease = pd.read_csv("EASE.csv").rename(
    columns={"user": "user_id", "item": "item_id"}
)
cdae = pd.read_csv("CDAE.csv").rename(
    columns={"user": "user_id", "item": "item_id"}
)

admm["rank_admm"] = admm.groupby("user_id").cumcount() + 1
ease["rank_ease"] = ease.groupby("user_id").cumcount() + 1
cdae["rank_cdae"] = cdae.groupby("user_id").cumcount() + 1

df = admm.merge(ease, on=["user_id", "item_id"], how="outer")
df = df.merge(cdae, on=["user_id", "item_id"], how="outer")

MAX_RANK = 1000
df[["rank_admm", "rank_ease", "rank_cdae"]] = (
    df[["rank_admm", "rank_ease", "rank_cdae"]].fillna(MAX_RANK)
)

df["final_rank"] = (
    0.45 * df["rank_admm"]
    + 0.30 * df["rank_ease"]
    + 0.25 * df["rank_cdae"]
)

TOP_K = 10
df_final = (
    df
    .sort_values(["user_id", "final_rank"], ascending=[True, True])
    .groupby("user_id", sort=False)
    .head(TOP_K)
)

(
    df_final[["user_id", "item_id"]]
    .rename(columns={"user_id": "user", "item_id": "item"})
    .to_csv("ensemble_outputs/ensemble_03.csv", index=False)
)