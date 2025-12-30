import pandas as pd
import numpy as np
import torch
import random
import os
from tqdm import tqdm
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
from catboost import CatBoostRanker, Pool
from utils import SequenceGenerator

# ==========================================
# 1. 모델 로드 함수 (User 코드 반영)
# ==========================================
def load_recbole_model(saved_file):
    """
    저장된 .pth 파일에서 모델과 데이터셋 정보를 불러옵니다.
    """
    print(f"Loading model from {saved_file}...")
    # config, model, dataset, train_data, valid_data, test_data 순서로 반환됨
    config, model, dataset, _, _, _ = load_data_and_model(model_file=saved_file)
    model.eval()  # 평가 모드 필수
    return model, dataset, config

# ==========================================
# 2. 점수 추출 함수 (배치 처리로 속도 최적화)
# ==========================================
seq_gen = SequenceGenerator('user_history.pkl')  # 유저별 시청 이력 로드

def get_model_scores(model, dataset, user_list, item_list, batch_size=2048):
    """
    리스트로 된 유저, 아이템에 대해 모델의 예측 점수를 반환합니다.
    * max_seq_len은 50으로 고정
    """
    device = model.device
    total_scores = []
    
    # 모델이 sequential인지 확인
    is_sequential = hasattr(model, 'ITEM_SEQ')

    temp_config = {
        'device': device,
        'MAX_ITEM_LIST_LENGTH': 50
    }

    # 배치 단위로 처리 (한 번에 다 넣으면 메모리 터질 수 있음)
    for i in tqdm(range(0, len(user_list), batch_size), desc=f"Scoring {model.__class__.__name__}"):
        batch_users = user_list[i : i+batch_size]
        batch_items = item_list[i : i+batch_size]
        
        # 텐서 변환
        user_tensor = torch.tensor(batch_users).to(device)
        item_tensor = torch.tensor(batch_items).to(device)
        
        # Interaction 객체 생성
        interaction = {
            'user_id': user_tensor,
            'item_id': item_tensor
        }

        # 시퀀셜 모델인 경우 시청 이력 추가
        if is_sequential:
            item_seq, item_len = seq_gen.get_input_for_model(dataset, batch_users, temp_config)

            interaction[model.ITEM_SEQ] = item_seq
            interaction[model.ITEM_SEQ_LEN] = item_len

        # Interaction 객체 생성
        interaction = Interaction(interaction)
        
        # 점수 예측
        with torch.no_grad():
            scores = model.predict(interaction)
            total_scores.extend(scores.cpu().numpy())
            
    return np.array(total_scores)

# ==========================================
# 3. 학습 데이터 생성 (Negative Sampling)
# ==========================================
def generate_training_data(train_csv_path, dataset, num_neg=2, max_pos=150):
    """
    실제 시청 기록(Pos)과 안 본 영화(Neg)를 섞어서 학습 데이터를 만듭니다.
    """
    print("Generating Training Data (Pos + Neg, Max Pos={max_pos})...")
    
    # 1. 원본 학습 데이터 로드 (User ID, Item ID가 문자열일 수 있음)
    origin_df = pd.read_csv(train_csv_path)
    
    # 2. ID 매핑 (문자열 -> RecBole 내부 숫자 ID)
    # 데이터셋에 없는 ID가 들어오면 에러가 나므로, 있는 것만 필터링하거나 주의 필요
    # 여기서는 train 데이터이므로 dataset에 다 있다고 가정합니다.
    origin_df['user_id_idx'] = origin_df['user'].map(lambda x: dataset.token2id(dataset.uid_field, str(x)))
    origin_df['item_id_idx'] = origin_df['item'].map(lambda x: dataset.token2id(dataset.iid_field, str(x)))
    
    # 유저가 본 아이템 목록 (Negative 뽑을 때 제외용)
    user_seen = origin_df.groupby('user_id_idx')['item_id_idx'].apply(set).to_dict()
    
    users, items, targets = [], [], []
    
    # 3. 데이터 생성 루프
    unique_users = origin_df['user_id_idx'].unique()
    
    for u in tqdm(unique_users, desc="Sampling Negatives"):
        seen_items = list(user_seen.get(u, set()))
        
        if len(seen_items) > max_pos:
            seen_items = seen_items[-max_pos:]  # 최근 max_pos개만 사용
        
        # (1) Positive Data (본 거) -> Target 1
        for i in seen_items:
            users.append(u)
            items.append(i)
            targets.append(1)
            
        # (2) Negative Data (안 본 거) -> Target 0
        # Positive 개수 * num_neg 만큼 뽑기
        num_to_sample = len(seen_items) * num_neg
        
        # 안 본 것들 중에서 랜덤 추출
        # (set 연산은 느릴 수 있으니 단순하게 random으로 뽑고 seen인지 체크하는 게 빠를 수 있음)
        count = 0
        while count < num_to_sample:
            rand_item = random.randint(1, dataset.item_num - 1)
            if rand_item not in seen_items:
                users.append(u)
                items.append(rand_item)
                targets.append(0)
                count += 1
                
    return pd.DataFrame({'user': users, 'item': items, 'target': targets})

# ==========================================
# 메인 실행 코드
# ==========================================
def main():
    # -----------------------------------------------------------
    # [설정] 파일 경로들을 본인 환경에 맞게 수정하세요!
    # -----------------------------------------------------------
    SASREC_PATH = 'saved/SASRec-best.pth'
    LIGHTGCN_PATH = 'saved/LightGCN-best.pth'
    EASE_PATH = 'saved/EASE-best.pth' # RecBole로 학습한 EASE라고 가정

    TRAIN_CSV = '../../data/train/train_ratings.csv'       # 원본 학습 데이터 (정답지)
    TOP100_CSV = 'top100.csv'  # 추론할 후보군 (Top-100)
    OUTPUT_CSV = '../../data/eval/final_submission.csv'   # 최종 결과 파일
    
    # 중간 저장 파일명 정의
    TRAIN_WITH_SCORES_PATH = 'train_data_with_scores.csv'
    CANDIDATES_WITH_SCORES_PATH = 'candidates_with_scores.csv'

    # -----------------------------------------------------------
    # 1. 모델 및 데이터셋 로드
    # -----------------------------------------------------------
    # 데이터셋 정보(ID 매핑)는 하나만 있어도 되므로 SASRec거를 메인으로 씁니다.
    if os.path.exists(TRAIN_WITH_SCORES_PATH):
        print(f"✅ 이미 계산된 학습 데이터가 있습니다! 로드 중... ({TRAIN_WITH_SCORES_PATH})")
        train_df = pd.read_csv(TRAIN_WITH_SCORES_PATH)
    else:
        print("🚀 학습 데이터 및 점수 계산을 시작합니다...")

        sas_model, dataset, _ = load_recbole_model(SASREC_PATH)
        lgcn_model, _, _ = load_recbole_model(LIGHTGCN_PATH)
        ease_model, _, _ = load_recbole_model(EASE_PATH)
    
    # -----------------------------------------------------------
    # 2. CatBoost 학습용 데이터 생성 (Target 0, 1)
    # -----------------------------------------------------------
        train_df = generate_training_data(TRAIN_CSV, dataset, num_neg=2)
    
    # -----------------------------------------------------------
    # 3. 학습 데이터에 대한 모델 점수 계산 (Feature Engineering)
    # -----------------------------------------------------------
        user_ids = train_df['user'].values
        item_ids = train_df['item'].values
    
        train_df['sasrec_score'] = get_model_scores(sas_model, dataset, user_ids, item_ids)
        train_df['lightgcn_score'] = get_model_scores(lgcn_model, dataset, user_ids, item_ids)
        train_df['ease_score'] = get_model_scores(ease_model, dataset, user_ids, item_ids)
    
    # (선택) 여기에 장르, 감독 등 Side Info가 있다면 merge 하세요!
    # train_df = pd.merge(train_df, genre_df, left_on='item', right_on='item_idx', how='left')
        train_df.to_csv(TRAIN_WITH_SCORES_PATH, index=False)
        print("CatBoost 학습 데이터 준비 완료!")
        print(train_df.head())

    # -----------------------------------------------------------
    # 4. CatBoostRanker 학습
    # -----------------------------------------------------------
    # 랭킹 학습을 위해 유저별로 정렬
    train_df.sort_values(by='user', inplace=True)
    
    train_pool = Pool(
        data=train_df[['sasrec_score', 'lightgcn_score', 'ease_score']], # + Side Info
        label=train_df['target'],
        group_id=train_df['user'] # 같은 유저끼리 그룹핑
    )
    
    model = CatBoostRanker(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='YetiRank',
        eval_metric='RecallAt:top=10',
        task_type="GPU", # GPU 있으면 "GPU"로 변경
        verbose=100,
        early_stopping_rounds=50,
        random_seed=42
    )
    
    print("Training CatBoost...")
    model.fit(train_pool)
    
    # -----------------------------------------------------------
    # 5. 최종 추론 (Top-100 후보군 사용)
    # -----------------------------------------------------------
    if os.path.exists(CANDIDATES_WITH_SCORES_PATH):
        print(f"✅ 이미 계산된 후보군 데이터가 있습니다! 로드 중... ({CANDIDATES_WITH_SCORES_PATH})")
        candidates = pd.read_csv(CANDIDATES_WITH_SCORES_PATH)
    else:
        print("🚀 후보군 데이터에 대한 점수 계산을 시작합니다...")
    
        # dataset 변수가 없으면 모델을 로드
        # (위에서 학습 데이터 로드할 때 모델 로딩을 건너뛰었을 경우를 대비함)
        if 'dataset' not in locals():
            print("⚠️ 모델과 데이터셋이 메모리에 없어서 다시 로드합니다...")
            sas_model, dataset, sas_config = load_recbole_model(SASREC_PATH) # config도 같이 받기
            lgcn_model, _, _ = load_recbole_model(LIGHTGCN_PATH)
            ease_model, _, _ = load_recbole_model(EASE_PATH)
            
        # user_history 다시 로드
        if 'seq_gen' not in locals():
            print("Loading SequenceGenerator from pkl...")
            global seq_gen
            seq_gen = SequenceGenerator('user_history.pkl')
    
        candidates = pd.read_csv(TOP100_CSV) 
        # candidates에는 user, item (원본 ID)이 있다고 가정
    
        # ID 변환 (문자열 -> 숫자)
        candidates['user_idx'] = candidates['user'].map(lambda x: dataset.token2id(dataset.uid_field, str(x)))
        candidates['item_idx'] = candidates['item'].map(lambda x: dataset.token2id(dataset.iid_field, str(x)))

        # 점수 계산 (만약 CSV에 점수가 없다면 계산, 있으면 생략 가능)
        c_users = candidates['user_idx'].values
        c_items = candidates['item_idx'].values
    
        candidates['sasrec_score'] = get_model_scores(sas_model, dataset, c_users, c_items)
        candidates['lightgcn_score'] = get_model_scores(lgcn_model, dataset, c_users, c_items)
        candidates['ease_score'] = get_model_scores(ease_model, dataset, c_users, c_items)

        candidates.drop(columns=['user_idx', 'item_idx'], inplace=True, errors='ignore')  # 정수 ID 컬럼 제거

        candidates.to_csv(CANDIDATES_WITH_SCORES_PATH, index=False)
        print("후보군 점수 계산 완료!")
        print(candidates.head())

    # CatBoost 예측을 위해 정렬
    print("Predicting with CatBoost...")
    candidates.sort_values(by='user', inplace=True)
    
    test_pool = Pool(
        data=candidates[['sasrec_score', 'lightgcn_score', 'ease_score']],
        group_id=candidates['user']
    )
    
    # 최종 점수 예측
    candidates['final_score'] = model.predict(test_pool)
    
    # -----------------------------------------------------------
    # 6. Top-10 선정 및 저장
    # -----------------------------------------------------------
    print("Selecting Top-10...")
    top10 = candidates.sort_values(['user', 'final_score'], ascending=[True, False]) \
                      .groupby('user').head(10)
    
    # 필요한 컬럼만 저장
    top10[['user', 'item']].to_csv(OUTPUT_CSV, index=False)
    print(f"완료! {OUTPUT_CSV}에 저장되었습니다.")

if __name__ == "__main__":
    main()