# ensemble_hard_vote.py
import argparse
import os
import pandas as pd
import numpy as np

def hard_ensemble(df_list, k=50, topk=10, mode="count_borda"):
    """
    df_list: list of DataFrames with columns [user,item,rank]
    mode:
      - count_only
      - count_borda   (추천)
      - borda_only
      - count_rr
      - rr_only
    """
    df = pd.concat(df_list, ignore_index=True)

    # vote count
    df["vote"] = 1

    # rank-based scores
    df["borda"] = (k + 1) - df["rank"]          # 50->1, 1->50
    df["rr"] = 1.0 / df["rank"].astype(float)   # 1->1.0, 50->0.02

    agg = df.groupby(["user", "item"], as_index=False).agg(
        vote_count=("vote", "sum"),
        borda_sum=("borda", "sum"),
        rr_sum=("rr", "sum"),
        best_rank=("rank", "min"),
    )

    if mode == "count_only":
        agg["score"] = agg["vote_count"]
    elif mode == "borda_only":
        agg["score"] = agg["borda_sum"]
    elif mode == "rr_only":
        agg["score"] = agg["rr_sum"]
    elif mode == "count_rr":
        agg["score"] = agg["vote_count"] * 1000 + agg["rr_sum"]
    else:  # count_borda (default)
        agg["score"] = agg["vote_count"] * 1000 + agg["borda_sum"]

    # user별 topk
    agg = agg.sort_values(["user", "score", "best_rank"], ascending=[True, False, True])

    # 각 user마다 상위 topk 선택
    out = (
        agg.groupby("user", as_index=False)
           .head(topk)
           .loc[:, ["user", "item"]]
           .reset_index(drop=True)
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ease_csv", default="output/ease_top50.csv")
    ap.add_argument("--admm_csv", default="output/admm_top50.csv")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--mode", type=str, default="count_borda",
                    choices=["count_only","count_borda","borda_only","count_rr","rr_only"])
    ap.add_argument("--out", default="output/submission_ensemble.csv")
    args = ap.parse_args()

    ease = pd.read_csv(args.ease_csv)
    admm = pd.read_csv(args.admm_csv)

    # 기본 sanity check
    for name, d in [("ease", ease), ("admm", admm)]:
        if not set(["user","item","rank"]).issubset(d.columns):
            raise ValueError(f"{name} csv must have columns: user,item,rank")
        if d["rank"].min() < 1 or d["rank"].max() > args.k:
            raise ValueError(f"{name} rank out of range 1..{args.k}")

    sub = hard_ensemble([ease, admm], k=args.k, topk=args.topk, mode=args.mode)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print("Saved:", args.out, "rows=", len(sub))

if __name__ == "__main__":
    main()
