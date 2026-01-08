import pickle
import torch
import numpy as np
from recbole.data.interaction import Interaction

class SequenceGenerator:
    def __init__(self, history_file='user_history.pkl'):
        """저장된 시청 내역을 로드하여 메모리에 유지"""
        print(f"Loading history from {history_file}...")
        with open(history_file, 'rb') as f:
            self.raw_history = pickle.load(f)
            
    def get_input_for_model(self, dataset, user_list, config):
        """
        RecBole 모델 입력용 시퀀스 텐서 생성
        Args:
            dataset: 현재 모델의 Dataset (ID 매핑 정보 포함)
            user_list: 입력 유저 ID 리스트 (RecBole 내부 정수 ID)
            config: 모델 설정 (max_len 등)
        """
        device = config['device']
        max_len = config['MAX_ITEM_LIST_LENGTH']
        
        # 1. 내부 ID -> 원본 ID 복구 (매핑 테이블 활용)
        # 저장된 history 파일은 원본 ID(문자열)로 되어 있기 때문
        raw_user_ids = dataset.id2token(dataset.uid_field, user_list)
        
        seq_list = []
        len_list = []
        
        # 2. 시퀀스 변환 (원본 ID -> 현재 모델의 내부 ID)
        for raw_uid in raw_user_ids:
            raw_items = self.raw_history.get(raw_uid, [])
            
            # 원본 아이템 ID들을 현재 모델의 내부 정수 ID로 변환
            # (해당 모델 데이터셋에 없는 아이템은 0번 처리 또는 제외)
            try:
                item_indices = dataset.token2id(dataset.iid_field, raw_items)
            except:
                # 안전장치: 일부 아이템이 없을 경우 필터링
                item_indices = [
                    dataset.token2id(dataset.iid_field, x) 
                    for x in raw_items if x in dataset.field2token_id[dataset.iid_field]
                ]
            
            # 3. Truncation (길이 자르기) & Padding (0 채우기)
            # SASRec은 보통 '현재 시점'을 예측하므로 마지막 것까지 포함해서 입력으로 씀
            seq = item_indices[-max_len:]
            length = len(seq)
            
            # Pre-padding (앞에 0 채우기)
            pad_len = max_len - length
            seq = [0] * pad_len + list(seq)
            
            seq_list.append(seq)
            len_list.append(length)
            
        # 4. 텐서 변환
        return (torch.LongTensor(seq_list).to(device), 
                torch.LongTensor(len_list).to(device))