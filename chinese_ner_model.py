#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文名片NER模型定義
包含模型架構和數據集類
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import List, Dict, Any

class ChineseNERModel(nn.Module):
    """中文名片NER模型"""
    
    def __init__(self, model_name: str = "hfl/chinese-roberta-wwm-ext", num_labels: int = 21):
        super(ChineseNERModel, self).__init__()
        
        # BERT基礎模型
        self.bert = BertModel.from_pretrained(model_name)
        
        # 獲取BERT的隱藏層維度
        self.hidden_size = self.bert.config.hidden_size
        
        # 分類層
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # 標籤映射
        self.label2id = {
            'O': 0, 'B-NAME': 1, 'I-NAME': 2, 'B-TITLE': 3, 'I-TITLE': 4,
            'B-PHONE': 5, 'I-PHONE': 6, 'B-FAX': 7, 'I-FAX': 8,
            'B-MOBILE': 9, 'I-MOBILE': 10, 'B-COMPANY': 11, 'I-COMPANY': 12,
            'B-EMAIL': 13, 'I-EMAIL': 14, 'B-ADDRESS': 15, 'I-ADDRESS': 16,
            'B-TAX_ID': 17, 'I-TAX_ID': 18, 'B-WEBSITE': 19, 'I-WEBSITE': 20
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向傳播"""
        # BERT編碼
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 獲取最後一層的隱藏狀態
        sequence_output = outputs[0]
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 分類
        logits = self.classifier(sequence_output)
        
        return logits
    
    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        """預測標籤"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            predictions = torch.argmax(logits, dim=-1)
        return predictions

class BusinessCardDataset(Dataset):
    """中文名片數據集"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer: BertTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 標籤映射
        self.label2id = {
            'O': 0, 'B-NAME': 1, 'I-NAME': 2, 'B-TITLE': 3, 'I-TITLE': 4,
            'B-PHONE': 5, 'I-PHONE': 6, 'B-FAX': 7, 'I-FAX': 8,
            'B-MOBILE': 9, 'I-MOBILE': 10, 'B-COMPANY': 11, 'I-COMPANY': 12,
            'B-EMAIL': 13, 'I-EMAIL': 14, 'B-ADDRESS': 15, 'I-ADDRESS': 16,
            'B-TAX_ID': 17, 'I-TAX_ID': 18, 'B-WEBSITE': 19, 'I-WEBSITE': 20
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除batch維度
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0)
        
        # 處理標籤
        label_ids = self._align_labels(label, input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def _align_labels(self, labels: List[str], input_ids: torch.Tensor) -> List[int]:
        """對齊標籤與token"""
        label_ids = []
        word_ids = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # 將字符級標籤轉換為詞級標籤
        word_labels = []
        char_idx = 0
        
        for word_id in word_ids:
            if word_id is None or word_id in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                word_labels.append(-100)  # 忽略標籤
            elif word_id.startswith('##'):
                # 子詞，使用前一個詞的標籤
                if word_labels and word_labels[-1] != -100:
                    word_labels.append(word_labels[-1])
                else:
                    word_labels.append(0)  # O標籤
            else:
                # 新詞，取第一個字符的標籤
                if char_idx < len(labels):
                    word_labels.append(self.label2id.get(labels[char_idx], 0))
                    char_idx += 1
                else:
                    word_labels.append(0)  # O標籤
        
        return word_labels[:len(input_ids)]

def create_model(model_name: str = "hfl/chinese-roberta-wwm-ext") -> ChineseNERModel:
    """創建模型實例"""
    return ChineseNERModel(model_name=model_name)

def load_model(model_path: str, model_name: str = "hfl/chinese-roberta-wwm-ext") -> ChineseNERModel:
    """加載訓練好的模型"""
    model = ChineseNERModel(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def predict_text(model: ChineseNERModel, tokenizer: BertTokenizer, text: str, device: str = 'cpu') -> List[str]:
    """預測單個文本的標籤"""
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    # 移動到指定設備
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # 預測
    with torch.no_grad():
        logits = model(**encoding)
        predictions = torch.argmax(logits, dim=-1)
    
    # 轉換為標籤
    pred_labels = []
    word_ids = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    for i, (word_id, pred_id) in enumerate(zip(word_ids, predictions[0])):
        if word_id not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            if not word_id.startswith('##'):
                pred_labels.append(model.id2label.get(pred_id.item(), 'O'))
    
    return pred_labels

def extract_entities(text: str, labels: List[str]) -> Dict[str, str]:
    """從文本和標籤中提取實體"""
    entities = {
        'name': '',
        'title': '',
        'phone': '',
        'mobile': '',
        'company': '',
        'email': '',
        'address': '',
        'tax_id': '',
        'website': ''
    }
    
    current_entity = None
    current_text = ""
    
    for char, label in zip(text, labels):
        if label.startswith('B-'):
            # 保存之前的實體
            if current_entity and current_text:
                entity_type = current_entity.replace('B-', '').replace('I-', '').lower()
                if entity_type in entities:
                    entities[entity_type] = current_text.strip()
            
            # 開始新實體
            current_entity = label
            current_text = char
        elif label.startswith('I-') and current_entity and label.replace('I-', '') == current_entity.replace('B-', ''):
            # 繼續當前實體
            current_text += char
        else:
            # 保存之前的實體
            if current_entity and current_text:
                entity_type = current_entity.replace('B-', '').replace('I-', '').lower()
                if entity_type in entities:
                    entities[entity_type] = current_text.strip()
            
            # 重置
            current_entity = None
            current_text = ""
    
    # 保存最後一個實體
    if current_entity and current_text:
        entity_type = current_entity.replace('B-', '').replace('I-', '').lower()
        if entity_type in entities:
            entities[entity_type] = current_text.strip()
    
    return entities 