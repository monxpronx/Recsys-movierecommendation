import argparse
import os
import pandas as pd
import torch

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

DEFAULT_TOPK = 100
BATCH_SIZE = 1024  # GPU 메모리에 맞게 조정
OUTPUT_DIR = "output"


def make_output_path(checkpoint_path: str, topk: int) -> str:
    base = os.path.basename(checkpoint_path)
    name, _ = os.path.splitext(base)
    # 파일명에 topk 포함
    return os.path.join(OUTPUT_DIR, f"submission_{name}_top{topk}.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    # 모델 및 데이터 로드
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        args.checkpoint
    )

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    # 테스트 유저 추출
    test_uids = test_data.dataset.inter_feat[uid_field].unique().tolist()
    test_uids = sorted(test_uids)

    model.eval()

    all_topk_score, all_topk_iid = [], []

    print(f"Start inference... (Total Users: {len(test_uids)}, Top-K: {args.topk})")

    # 배치 단위 추론
    for i in range(0, len(test_uids), BATCH_SIZE):
        batch_uids = test_uids[i : i + BATCH_SIZE]

        batch_score, batch_iid = full_sort_topk(
            batch_uids,
            model,
            test_data,
            k=args.topk,
            device=config["device"],
        )

        all_topk_score.append(batch_score)
        all_topk_iid.append(batch_iid)

        # 메모리 여유 확보
        torch.cuda.empty_cache()

    # 최종 합치기
    topk_score = torch.cat(all_topk_score, dim=0)
    topk_iid = torch.cat(all_topk_iid, dim=0)

    # 유저 ID 토큰 변환
    user_tokens = dataset.id2token(uid_field, test_uids)

    rows = []
    # 결과 추출 및 리스트 작성
    for idx, u in enumerate(user_tokens):
        # 1. 아이템 ID 변환 (Inner ID → Token)
        item_inner_ids = topk_iid[idx].cpu().numpy().tolist()
        item_tokens = dataset.id2token(iid_field, item_inner_ids)

        # 2. Score 추출 (Tensor -> Float)
        scores = topk_score[idx].cpu().numpy().tolist()

        # 3. (User, Item, Score) 형태로 저장
        for it, sc in zip(item_tokens, scores):
            rows.append((u, it, sc))

    # DataFrame 생성
    out_df = pd.DataFrame(rows, columns=["user", "item", "score"])

    # ID 타입 변환
    for col in ["user", "item"]:
        try:
            out_df[col] = out_df[col].astype(int)
        except Exception:
            pass

    # 결과 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = make_output_path(args.checkpoint, args.topk)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] saved → {out_path}")
    print(f"users={len(test_uids)} | rows={len(out_df)} | columns={list(out_df.columns)}")


if __name__ == "__main__":
    main()
