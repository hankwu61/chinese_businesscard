#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文名片NER模型測試腳本
"""

import os
import sys
from data_processor import BusinessCardDataProcessor
from chinese_ner_model import ChineseNERModel, BusinessCardDataset
from transformers import BertTokenizer

def test_data_generation():
    """測試數據生成"""
    print("=== 測試數據生成 ===")
    
    processor = BusinessCardDataProcessor()
    
    # 生成少量測試數據
    texts, labels = processor.generate_dataset(num_samples=5)
    
    print(f"生成了 {len(texts)} 個樣本")
    
    for i, (text, label) in enumerate(zip(texts, labels)):
        print(f"\n樣本 {i+1}:")
        print(f"文本: {text}")
        print(f"標籤: {label[:50]}...")  # 只顯示前50個標籤
    
    return texts, labels

def test_model_initialization():
    """測試模型初始化"""
    print("\n=== 測試模型初始化 ===")
    
    try:
        model = ChineseNERModel()
        print("✓ 模型初始化成功")
        
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        print("✓ Tokenizer加載成功")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ 模型初始化失敗: {e}")
        return None, None

def test_dataset_creation(texts, labels, tokenizer):
    """測試數據集創建"""
    print("\n=== 測試數據集創建 ===")
    
    try:
        dataset = BusinessCardDataset(texts, labels, tokenizer)
        print(f"✓ 數據集創建成功，大小: {len(dataset)}")
        
        # 測試一個樣本
        sample = dataset[0]
        print(f"✓ 樣本格式正確:")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  labels shape: {sample['labels'].shape}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ 數據集創建失敗: {e}")
        return None

def test_model_forward_pass(model, dataset):
    """測試模型前向傳播"""
    print("\n=== 測試模型前向傳播 ===")
    
    try:
        sample = dataset[0]
        
        # 添加batch維度
        input_ids = sample['input_ids'].unsqueeze(0)
        attention_mask = sample['attention_mask'].unsqueeze(0)
        token_type_ids = sample['token_type_ids'].unsqueeze(0)
        
        # 前向傳播
        import torch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
        
        print(f"✓ 前向傳播成功")
        print(f"  輸出shape: {outputs.shape}")
        print(f"  預期shape: (1, {input_ids.shape[1]}, 21)")
        
        return True
        
    except Exception as e:
        print(f"✗ 前向傳播失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("開始測試中文名片NER模型...\n")
    
    # 測試數據生成
    texts, labels = test_data_generation()
    
    # 測試模型初始化
    model, tokenizer = test_model_initialization()
    if model is None or tokenizer is None:
        print("模型初始化失敗，停止測試")
        return
    
    # 測試數據集創建
    dataset = test_dataset_creation(texts, labels, tokenizer)
    if dataset is None:
        print("數據集創建失敗，停止測試")
        return
    
    # 測試模型前向傳播
    success = test_model_forward_pass(model, dataset)
    
    if success:
        print("\n=== 所有測試通過！ ===")
        print("模型可以正常使用，可以開始訓練了。")
    else:
        print("\n=== 測試失敗 ===")
        print("請檢查錯誤信息並修復問題。")

if __name__ == "__main__":
    main() 