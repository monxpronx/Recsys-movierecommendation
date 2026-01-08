import argparse
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm # 진행상황바 표시

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

# ================= Configuration =================
DEFAULT_TOPK = 100
BATCH_SIZE = 1  # [중요] Jagged Tensor 문제를 피하기 위해 1명씩 처리합니다.
OUTPUT_DIR = "output"
# =================================================

def make_output_path(checkpoint_path: str, topk: int) -> str:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    base = os.path.basename(checkpoint_path)
    name, _ = os.path.splitext(base)
    return os.path.join(OUTPUT_DIR, f"submission_{name}_top{topk}.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        args.checkpoint
    )
    
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    target_uids = np.arange(1, dataset.user_num)
    print(f"Start inference... (Total Users: {len(target_uids)})")
    
    model.eval()

    # 결과를 저장할 리스트
    final_user_ids = []
    final_item_ids = []
    final_scores = []

    # 2. Inference Loop (One by One)
    with torch.no_grad():
        # tqdm을 사용하여 진행률을 시각적으로 확인합니다.
        for uid in tqdm(target_uids, desc="Processing Users"):
            # (1) 유저 1명 선택
            # full_sort_topk는 텐서 입력을 기대하므로 1개짜리 텐서 생성
            # CPU Tensor로 생성 (내부에서 device 처리됨)
            batch_uids_torch = torch.tensor([uid])

            # (2) Top-K 예측
            # output shape: [Seq_Len, K] (1명분이므로 Batch차원이 없거나 1임)
            batch_scores, batch_iids = full_sort_topk(
                batch_uids_torch, 
                model, 
                test_data, 
                k=args.topk, 
                device=config["device"]
            )

            # (3) [핵심] 마지막 시점(Last Step) 추출
            # BERT4Rec은 시퀀스 전체에 대한 예측을 내놓을 수 있음.
            # 우리는 무조건 '가장 마지막' 예측이 타겟임.
            
            # 차원이 3차원인 경우 [1, Seq_Len, K] -> [Seq_Len, K]로 변경
            if batch_iids.dim() == 3:
                batch_iids = batch_iids.squeeze(0)
                batch_scores = batch_scores.squeeze(0)
            
            # 마지막 줄(Last Row) 가져오기
            last_iids = batch_iids[-1]     # Shape: [K]
            last_scores = batch_scores[-1] # Shape: [K]

            # (4) GPU -> CPU -> Numpy
            last_iids = last_iids.cpu().numpy()
            last_scores = last_scores.cpu().numpy()

            # (5) 결과 저장
            # 유저 ID를 K번 반복해서 저장
            final_user_ids.extend([uid] * args.topk)
            final_item_ids.extend(last_iids)
            final_scores.extend(last_scores)

    print("Inference done. Converting IDs to tokens...")

    # 3. Save
    # numpy 변환
    final_user_ids = np.array(final_user_ids).astype(int)
    final_item_ids = np.array(final_item_ids).astype(int)
    final_scores = np.array(final_scores)

    # ID 변환 (Internal -> Original)
    original_users = dataset.id2token(uid_field, final_user_ids)
    original_items = dataset.id2token(iid_field, final_item_ids)

    out_df = pd.DataFrame({
        "user_id": original_users,
        "item_id": original_items,
        "score": final_scores
    })

    out_path = make_output_path(args.checkpoint, args.topk)
    out_df.to_csv(out_path, index=False)

    print(f"[Success] Saved to {out_path}")

if __name__ == "__main__":
    main()