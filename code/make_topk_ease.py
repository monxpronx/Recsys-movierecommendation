# make_topk_ease.py
import argparse
import os
import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from utils import check_path, get_user_seqs, set_seed

class EASE:
    def __init__(self, reg_lambda=500.0):
        self.reg_lambda = float(reg_lambda)
        self.B = None

    def fit(self, X):
        if sp is not None and sp.issparse(X):
            G = (X.T @ X).toarray()
        else:
            G = X.T @ X

        diag = np.diag_indices(G.shape[0])
        G[diag] += self.reg_lambda
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[diag] = 0.0
        self.B = B

    def predict(self, X):
        return X @ self.B

def build_x_from_user_seq(user_seq, num_items):
    rows, cols = [], []
    for u, seq in enumerate(user_seq):
        for it in seq:
            rows.append(u)
            cols.append(it)
    data = np.ones(len(rows), dtype=np.float32)
    if sp is not None:
        return sp.csr_matrix((data, (rows, cols)), shape=(len(user_seq), num_items), dtype=np.float32)
    X = np.zeros((len(user_seq), num_items), dtype=np.float32)
    X[rows, cols] = 1.0
    return X

def mask_seen(scores, seen_matrix):
    if sp is not None and sp.issparse(seen_matrix):
        scores[seen_matrix.astype(bool).toarray()] = -np.inf
    else:
        scores[seen_matrix.astype(bool)] = -np.inf
    return scores

def topk_preds(scores, topk):
    preds = []
    n_items = scores.shape[1]
    for u in range(scores.shape[0]):
        row = scores[u]
        k = min(topk, n_items)
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]
        preds.append(idx.tolist())
    return preds

def reindex_user_seq(user_seq_raw):
    uniq_items = set()
    for seq in user_seq_raw:
        for it in seq:
            if it > 0:
                uniq_items.add(it)
    uniq_items = np.array(sorted(uniq_items), dtype=np.int64)
    item2idx = {int(it): i for i, it in enumerate(uniq_items)}
    idx2item = uniq_items

    user_seq_idx = []
    for seq in user_seq_raw:
        user_seq_idx.append([item2idx[int(it)] for it in seq if it > 0 and int(it) in item2idx])
    return user_seq_idx, item2idx, idx2item

def build_seen_matrix_from_user_seq(user_seq_idx, num_items):
    rows, cols = [], []
    for u, seq in enumerate(user_seq_idx):
        for it in seq:
            rows.append(u); cols.append(it)
    data = np.ones(len(rows), dtype=np.float32)
    if sp is not None:
        return sp.csr_matrix((data, (rows, cols)), shape=(len(user_seq_idx), num_items), dtype=np.float32)
    M = np.zeros((len(user_seq_idx), num_items), dtype=np.float32)
    M[rows, cols] = 1.0
    return M

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--reg_lambda", type=float, default=500.0)
    parser.add_argument("--topk", type=int, default=50)  # ✅ Top-50
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    data_file = os.path.join(args.data_dir, "train_ratings.csv")

    # raw user/item
    raw = pd.read_csv(data_file)
    _, user_uniques = pd.factorize(raw["user"])  # ✅ ADMM과 동일하게 raw에서 factorize

    # sequences from utils
    user_seq_raw, max_item, _, _, _ = get_user_seqs(data_file)

    # reindex items (EASE 내부 학습용)
    user_seq_idx, item2idx, idx2item = reindex_user_seq(user_seq_raw)
    num_items = len(idx2item)

    X = build_x_from_user_seq(user_seq_idx, num_items)
    seen = build_seen_matrix_from_user_seq(user_seq_idx, num_items)

    model = EASE(reg_lambda=args.reg_lambda)
    model.fit(X)

    scores = model.predict(X)
    scores = mask_seen(scores, seen)

    preds_idx = topk_preds(scores, args.topk)  # reindexed item
    preds_orig = [[int(idx2item[i]) for i in row] for row in preds_idx]  # original item id

    # Top-50 rank CSV (long format)
    rows = []
    for u_idx, rec_items in enumerate(preds_orig):
        user_id = user_uniques[u_idx]
        for r, item_id in enumerate(rec_items, start=1):
            rows.append((user_id, item_id, r))

    df = pd.DataFrame(rows, columns=["user", "item", "rank"])
    out_path = os.path.join(args.output_dir, f"ease_top{args.topk}.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path, "rows=", len(df))

if __name__ == "__main__":
    main()
