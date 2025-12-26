# run_admm_slim_eval.py
import argparse
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from utils import (
    recall_at_k,
    precision_at_k,
    ndcg_k,
    mapk,
    generate_rating_matrix_valid,
    generate_rating_matrix_test,
    generate_rating_matrix_submission
    
)


from admm_slim import ADMMSLIM

DATA_DIR = "../data/train/"
DATA_FILE = DATA_DIR + "train_ratings.csv"

# ---------------------------
# GT 구성
# ---------------------------
def build_gt(user_seq, mode):
    gt = []
    for seq in user_seq:
        if mode == "valid":
            gt.append([seq[-2]] if len(seq) >= 2 else [])
        elif mode == "test":
            gt.append([seq[-1]] if len(seq) >= 1 else [])
        else:
            raise ValueError
    return gt


# ---------------------------
# 추천 + seen mask
# ---------------------------
def recommend_topk(X_hist, W, topk=10, batch_size=2048):
    X_hist = X_hist.tocsr()
    U, I = X_hist.shape
    preds = []

    for s in range(0, U, batch_size):
        e = min(U, s + batch_size)
        Xb = X_hist[s:e]
        scores = (Xb @ W).astype(np.float32)

        # seen item masking
        for i in range(e - s):
            u = s + i
            seen = X_hist.indices[X_hist.indptr[u]:X_hist.indptr[u + 1]]
            scores[i, seen] = -np.inf

        idx = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        topk_idx = np.take_along_axis(idx, order, axis=1)

        preds.extend(topk_idx.tolist())

    return preds


# ---------------------------
# 평가 루틴
# ---------------------------
def evaluate(model, user_seq, train_matrix, eval_matrix, mode, topk):
    model.fit(train_matrix)

    gt = build_gt(user_seq, mode)
    pred = recommend_topk(eval_matrix, model.W, topk=topk)

    return {
        f"Recall@{topk}": recall_at_k(gt, pred, topk),
        f"NDCG@{topk}": ndcg_k(gt, pred, topk),
        f"Precision@{topk}": precision_at_k(gt, pred, topk),
        f"MAP@{topk}": mapk(gt, pred, topk),
    }

def get_user_seqs(data_file):
    rating_df = pd.read_csv(data_file)

    # ✅ (중요) user/item을 0..n-1 연속 인덱스로 리매핑
    rating_df["user"], user_uniques = pd.factorize(rating_df["user"])
    rating_df["item"], item_uniques = pd.factorize(rating_df["item"])

    num_users = rating_df["user"].nunique()
    num_items = rating_df["item"].nunique()

    # user별 item 시퀀스
    lines = rating_df.sort_values(["user"]).groupby("user")["item"].apply(list)
    user_seq = lines.tolist()

    # 기존 함수 재사용 (shape는 nunique 기반으로 정확)
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix  = generate_rating_matrix_test(user_seq, num_users, num_items)
    submission_rating_matrix = generate_rating_matrix_submission(user_seq, num_users, num_items)

    # 기존 반환 형태 맞춤: max_item 대신 num_items-1 같은 의미 값 반환
    max_item = num_items - 1

    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )
# ---------------------------
# main
# ---------------------------
def main(args):
    user_seq, max_item, valid_matrix, test_matrix, _ = get_user_seqs(DATA_FILE)

    print("Users:", len(user_seq), "Items:", max_item)

    model = ADMMSLIM(
        l1=args.l1,
        l2=args.l2,
        rho=args.rho,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print("\n[VALID EVAL]")
    valid_res = evaluate(
        model,
        user_seq,
        train_matrix=valid_matrix,
        eval_matrix=valid_matrix,
        mode="valid",
        topk=args.topk,
    )
    for k, v in valid_res.items():
        print(f"{k}: {v:.6f}")

    model = ADMMSLIM(
        l1=args.l1,
        l2=args.l2,
        rho=args.rho,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print("\n[TEST EVAL]")
    test_res = evaluate(
        model,
        user_seq,
        train_matrix=test_matrix,
        eval_matrix=test_matrix,
        mode="test",
        topk=args.topk,
    )
    for k, v in test_res.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()
    main(args)
