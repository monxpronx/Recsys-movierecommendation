# make_topk_admm.py
import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from run_admm_slim_eval import get_user_seqs
from admm_slim import ADMMSLIM

DATA_DIR = "../data/train/"
DATA_FILE = os.path.join(DATA_DIR, "train_ratings.csv")
OUT_DIR = "output"

def recommend_topk_with_score(X_hist: sp.csr_matrix, W, topk=50, batch_size=2048):
    X_hist = X_hist.tocsr()
    U, I = X_hist.shape

    all_items = []
    all_scores = []

    for s in range(0, U, batch_size):
        e = min(U, s + batch_size)
        Xb = X_hist[s:e]
        scores = (Xb @ W).astype(np.float32)

        # seen mask
        for i in range(e - s):
            u = s + i
            seen = X_hist.indices[X_hist.indptr[u]:X_hist.indptr[u + 1]]
            scores[i, seen] = -np.inf

        # Top-K
        idx = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)

        topk_idx = np.take_along_axis(idx, order, axis=1)
        topk_score = np.take_along_axis(part, order, axis=1)

        all_items.extend(topk_idx.tolist())
        all_scores.extend(topk_score.tolist())

    return all_items, all_scores


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    user_seq, max_item, valid_matrix, test_matrix, submission_matrix = get_user_seqs(DATA_FILE)

    raw = pd.read_csv(DATA_FILE)
    _, user_uniques = pd.factorize(raw["user"])
    _, item_uniques = pd.factorize(raw["item"])

    X_train = submission_matrix

    model = ADMMSLIM(l1=args.l1, l2=args.l2, rho=args.rho, max_iter=args.max_iter, tol=args.tol)
    print("Fitting ADMM-SLIM...")
    model.fit(X_train)

    print("Predicting TopK...")
    preds, scores = recommend_topk_with_score(X_train, model.W, topk=args.topk, batch_size=args.batch_size)
    # Top-50 rank CSV (long format)
    rows = []
    for u_idx, rec_items in enumerate(preds):
        user_id = user_uniques[u_idx]
        for r, it in enumerate(rec_items, start=1):
            item_id = item_uniques[it]
            sc = scores[u_idx][r - 1]
            rows.append((user_id, item_id, r, float(sc)))

    df = pd.DataFrame(rows, columns=["user", "item", "rank", "score"])
    out_path = os.path.join(OUT_DIR, f"admm_top{args.topk}.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path, "rows=", len(df))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--l1", type=float, default=3.0)
    p.add_argument("--l2", type=float, default=200.0)
    p.add_argument("--rho", type=float, default=16000.0)
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--topk", type=int, default=50)         # ✅ Top-50
    p.add_argument("--batch_size", type=int, default=2048)
    args = p.parse_args()
    main(args)
