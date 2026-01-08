# make_submission.py
import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from run_admm_slim_eval import get_user_seqs  # ✅ 리매핑 버전( factorize )으로 교체된 get_user_seqs 여야 함
from admm_slim import ADMMSLIM


DATA_DIR = "../data/train/"
DATA_FILE = os.path.join(DATA_DIR, "train_ratings.csv")
OUT_DIR = "output"
OUT_FILE = os.path.join(OUT_DIR, "submission.csv")


def recommend_topk_submit(X_hist: sp.csr_matrix, W, topk=10, batch_size=2048):
    """
    X_hist: (U x I) user-item history (전체 시청/구매/클릭)
    W:      (I x I) item-item matrix
    return: preds (U x topk)  추천 item index (리매핑 인덱스 기준)
    """
    X_hist = X_hist.tocsr()
    U, I = X_hist.shape
    preds = []

    for s in range(0, U, batch_size):
        e = min(U, s + batch_size)
        Xb = X_hist[s:e]
        scores = (Xb @ W).astype(np.float32)

        # seen mask
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


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ✅ 여기서 중요한 건 get_user_seqs가 리매핑을 수행하는 버전이어야 함
    # 반환값: user_seq, max_item, valid_matrix, test_matrix, submission_matrix
    user_seq, max_item, valid_matrix, test_matrix, submission_matrix = get_user_seqs(DATA_FILE)

    num_users = submission_matrix.shape[0]
    num_items = submission_matrix.shape[1]
    print(f"Users: {num_users}, Items: {num_items}")

    # 유저 원본 ID(제출용) 확보: 리매핑 전 user를 그대로 쓰려면 원본 user 목록 필요
    # 가장 안전: 원본 파일에서 user unique 순서를 "factorize 순서"와 동일하게 맞춰야 함
    # utils에서 factorize를 썼다면, 같은 파일을 같은 방식으로 factorize해야 일치함
    raw = pd.read_csv(DATA_FILE)
    raw_users, user_uniques = pd.factorize(raw["user"])
    raw_items, item_uniques = pd.factorize(raw["item"])

    # 학습 데이터: 제출은 "전체 히스토리"로 학습하는 게 일반적으로 유리
    X_train = submission_matrix

    model = ADMMSLIM(
        l1=args.l1,
        l2=args.l2,
        rho=args.rho,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print("Fitting model on full data...")
    model.fit(X_train)

    print("Generating recommendations...")
    preds = recommend_topk_submit(X_train, model.W, topk=args.topk, batch_size=args.batch_size)

    # ✅ 리매핑 인덱스 -> 원래 item id로 복원
    # item_uniques는 "리매핑 인덱스 i -> 원래 item 값" 매핑 배열임
    result = []
    for u_idx, rec_items in enumerate(preds):
        user_id = user_uniques[u_idx]  # 원본 user id
        for it in rec_items:
            item_id = item_uniques[it]  # 원본 item id
            result.append((user_id, item_id))

    sub = pd.DataFrame(result, columns=["user", "item"])
    sub.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}  (rows={len(sub)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--l1", type=float, default=3.0)
    p.add_argument("--l2", type=float, default=200.0)
    p.add_argument("--rho", type=float, default=8000.0)
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2048)
    args = p.parse_args()
    main(args)
