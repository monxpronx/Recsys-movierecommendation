import argparse
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm # 진행상황바 표시

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

# ================= Configuration =================
DEFAULT_TOPK = 10
BATCH_SIZE = 1  # [중요] 길이 불일치 및 메모리 문제를 피하기 위해 1명씩 처리
OUTPUT_DIR = "output"
# =================================================

def make_output_path(checkpoint_path: str, topk: int) -> str:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    base = os.path.basename(checkpoint_path)
    name, _ = os.path.splitext(base)
    # GRU4RecF임을 파일명에 명시
    return os.path.join(OUTPUT_DIR, f"submission_{name}_top{topk}.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    # 1. Load Model & Data
    print(f"Loading model from {args.checkpoint}...")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        args.checkpoint
    )
    
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    # 전체 유저 대상 (1번 ~ 끝까지)
    target_uids = np.arange(1, dataset.user_num)
    print(f"Start inference... (Total Users: {len(target_uids)})")
    
    model.eval()

    # 결과를 저장할 리스트
    final_user_ids = []
    final_item_ids = []
    final_scores = []

    # 2. Inference Loop (One by One)
    with torch.no_grad():
        for uid in tqdm(target_uids, desc="Processing Users"):
            # (1) 유저 1명 선택
            # GRU4RecF는 Feature를 사용하지만, test_data 안에 이미 정보가 있으므로
            # 여기서는 User ID만 텐서로 넘겨주면 됩니다.
            batch_uids_torch = torch.tensor([uid])

            # (2) Top-K 예측
            # test_data를 넘겨줘야 side-info(genre, director 등)를 모델이 참조합니다.
            batch_scores, batch_iids = full_sort_topk(
                batch_uids_torch, 
                model, 
                test_data, 
                k=args.topk, 
                device=config["device"]
            )

            # (3) 차원 처리 (Safety Check)
            # GRU4RecF는 보통 [1, K]로 나오지만, 혹시 모를 3차원 출력([1, Seq, K])에 대비해
            # BERT4Rec 때와 동일한 안전장치를 둡니다.
            if batch_iids.dim() == 3:
                batch_iids = batch_iids.squeeze(0)
                batch_scores = batch_scores.squeeze(0)
            
            # 마지막 결과 추출
            # shape이 [1, K]일 때 [-1]을 하면 [K] (벡터)가 됩니다.
            last_iids = batch_iids[-1]
            last_scores = batch_scores[-1]

            # (4) GPU -> CPU -> Numpy
            last_iids = last_iids.cpu().numpy()
            last_scores = last_scores.cpu().numpy()

            # (5) 결과 저장
            # 유저 ID 1개를 K번 복사해서 리스트에 추가 (길이 맞춤의 핵심)
            final_user_ids.extend([uid] * args.topk)
            final_item_ids.extend(last_iids)
            final_scores.extend(last_scores)

    print("Inference done. Converting IDs to tokens...")

    # 3. Save
    # 리스트를 Numpy 배열로 변환
    final_user_ids = np.array(final_user_ids).astype(int)
    final_item_ids = np.array(final_item_ids).astype(int)
    final_scores = np.array(final_scores)

    # ID 변환 (Internal ID -> Original Token)
    # 이 과정이 있어야 리더보드 제출용 포맷(nm001 등)이 됩니다.
    original_users = dataset.id2token(uid_field, final_user_ids)
    original_items = dataset.id2token(iid_field, final_item_ids)

    # DataFrame 생성 (user, item, score 포함)
    out_df = pd.DataFrame({
        "user_id": original_users,
        "item_id": original_items,
        "score": final_scores
    })

    out_path = make_output_path(args.checkpoint, args.topk)
    out_df.to_csv(out_path, index=False)

    print(f"[Success] Saved to {out_path}")
    print(out_df.head()) # 결과 미리보기

if __name__ == "__main__":
    main()