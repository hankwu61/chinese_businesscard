#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
測試修復效果的腳本
"""

from data_processor import BusinessCardDataProcessor
from chinese_ner_model import BusinessCardDataset
from transformers import BertTokenizer

def test_data_generation():
    """測試數據生成和標籤對齊"""
    print("=== 測試數據生成和標籤對齊 ===\n")
    
    processor = BusinessCardDataProcessor()
    
    # 生成一個樣本
    text, labels = processor.generate_business_card_text()
    
    print(f"生成的名片文本:")
    print(f"  {text}")
    print(f"文本長度: {len(text)}")
    print(f"標籤數量: {len(labels)}")
    
    # 檢查標籤對齊
    print(f"\n標籤對齊檢查:")
    for i, (char, label) in enumerate(zip(text, labels)):
        if label != 'O':
            print(f"  位置 {i}: '{char}' -> {label}")
    
    return text, labels

def test_tokenizer_alignment():
    """測試分詞器和標籤對齊"""
    print("\n=== 測試分詞器和標籤對齊 ===\n")
    
    # 簡單測試案例
    test_text = "張三 總經理"
    test_labels = ['B-NAME', 'I-NAME', 'O', 'B-TITLE', 'I-TITLE', 'I-TITLE']
    
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    dataset = BusinessCardDataset([test_text], [test_labels], tokenizer)
    
    # 獲取第一個樣本
    sample = dataset[0]
    
    print(f"原始文本: {test_text}")
    print(f"原始標籤: {test_labels}")
    
    # 分詞
    tokens = tokenizer.tokenize(test_text)
    print(f"分詞結果: {tokens}")
    
    # 對齊標籤
    aligned_labels = dataset._align_labels(test_text, test_labels, tokens)
    print(f"對齊標籤: {aligned_labels}")
    
    # 轉換為ID
    label_ids = dataset._convert_labels_to_ids(aligned_labels)
    print(f"標籤ID: {label_ids}")
    
    return sample

def test_simple_prediction():
    """測試簡單預測"""
    print("\n=== 測試簡單預測 ===\n")
    
    # 創建一個簡單的測試案例
    test_cases = [
        ("張三", ['B-NAME', 'I-NAME']),
        ("總經理", ['B-TITLE', 'I-TITLE', 'I-TITLE']),
        ("02-1234-5678", ['B-PHONE'] + ['I-PHONE'] * 11),
    ]
    
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    
    for text, labels in test_cases:
        print(f"測試: {text}")
        print(f"標籤: {labels}")
        
        # 分詞
        tokens = tokenizer.tokenize(text)
        print(f"分詞: {tokens}")
        
        # 創建數據集
        dataset = BusinessCardDataset([text], [labels], tokenizer)
        sample = dataset[0]
        
        print(f"輸入ID shape: {sample['input_ids'].shape}")
        print(f"標籤 shape: {sample['labels'].shape}")
        print("-" * 50)

def main():
    """主測試函數"""
    print("測試修復效果...\n")
    
    # 測試數據生成
    text, labels = test_data_generation()
    
    # 測試分詞器對齊
    sample = test_tokenizer_alignment()
    
    # 測試簡單預測
    test_simple_prediction()
    
    print("\n測試完成！")

if __name__ == "__main__":
    main() 