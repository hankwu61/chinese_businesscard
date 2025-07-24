#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 chinese-roberta-wwm-ext 模型訓練中文名片NER識別器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
import os
from tqdm import tqdm

from chinese_ner_model import ChineseNERModel, BusinessCardDataset
from data_processor import BusinessCardDataProcessor

class RobertaNERTrainer:
    """使用RoBERTa模型的NER訓練器"""
    
    def __init__(self, model_name: str = "hfl/chinese-roberta-wwm-ext", device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = ChineseNERModel(model_name=model_name)
        self.model.to(self.device)
        
        # 標籤映射 - 10種實體類型
        self.label2id = {
            'O': 0, 'B-NAME': 1, 'I-NAME': 2, 'B-TITLE': 3, 'I-TITLE': 4,
            'B-PHONE': 5, 'I-PHONE': 6, 'B-FAX': 7, 'I-FAX': 8,
            'B-MOBILE': 9, 'I-MOBILE': 10, 'B-COMPANY': 11, 'I-COMPANY': 12,
            'B-EMAIL': 13, 'I-EMAIL': 14, 'B-ADDRESS': 15, 'I-ADDRESS': 16,
            'B-TAX_ID': 17, 'I-TAX_ID': 18, 'B-WEBSITE': 19, 'I-WEBSITE': 20
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        print(f"使用模型: {model_name}")
        print(f"設備: {self.device}")
        print(f"標籤數量: {len(self.label2id)}")
        
    def prepare_data(self, texts: list, labels: list, batch_size: int = 16):
        """準備數據"""
        dataset = BusinessCardDataset(texts, labels, self.tokenizer)
        
        # 分割訓練集和驗證集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"訓練樣本: {len(train_dataset)}")
        print(f"驗證樣本: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs: int = 5, learning_rate: float = 2e-5):
        """訓練模型"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略-100標籤
        
        # 學習率調度器
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        best_f1 = 0.0
        
        print(f"\n開始訓練，共 {epochs} 個epoch")
        print(f"學習率: {learning_rate}")
        print(f"總步數: {total_steps}")
        
        for epoch in range(epochs):
            print(f"\n{'='*20} Epoch {epoch + 1}/{epochs} {'='*20}")
            
            # 訓練階段
            self.model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc="Training")
            
            for batch in train_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                
                # 重塑輸出和標籤
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 驗證階段
            val_loss, val_metrics = self.evaluate(val_loader, criterion)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 'best_ner_model.pth')
                print("✅ 保存最佳模型 (F1提升)")
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_ner_model.pth')
                print("✅ 保存最佳模型 (Loss降低)")
    
    def evaluate(self, val_loader, criterion):
        """評估模型"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                val_loss += loss.item()
                
                # 預測
                preds = torch.argmax(outputs, dim=-1)
                
                # 收集預測和標籤
                for i in range(preds.shape[0]):
                    pred_seq = preds[i].cpu().numpy()
                    label_seq = labels[i].cpu().numpy()
                    mask = attention_mask[i].cpu().numpy()
                    
                    # 只保留非padding的標籤
                    pred_seq = pred_seq[mask == 1]
                    label_seq = label_seq[mask == 1]
                    
                    # 過濾掉-100標籤
                    valid_indices = label_seq != -100
                    pred_seq = pred_seq[valid_indices]
                    label_seq = label_seq[valid_indices]
                    
                    all_preds.extend(pred_seq)
                    all_labels.extend(label_seq)
        
        # 計算指標
        metrics = self.calculate_metrics(all_labels, all_preds)
        return val_loss / len(val_loader), metrics
    
    def calculate_metrics(self, true_labels, pred_labels):
        """計算評估指標"""
        # 轉換為標籤名稱
        true_label_names = [self.id2label[label] for label in true_labels]
        pred_label_names = [self.id2label[label] for label in pred_labels]
        
        # 計算分類報告
        report = classification_report(
            true_label_names, pred_label_names, 
            output_dict=True, zero_division=0
        )
        
        return {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }

def main():
    """主函數"""
    print("🚀 開始使用 chinese-roberta-wwm-ext 訓練中文名片NER模型")
    print("=" * 60)
    
    # 初始化數據處理器
    processor = BusinessCardDataProcessor()
    
    # 生成訓練數據
    print("📊 生成訓練數據...")
    texts, labels = processor.generate_dataset(num_samples=2000)  # 增加樣本數量
    
    print(f"📈 數據集大小: {len(texts)} 個樣本")
    
    # 保存數據集
    processor.save_dataset(texts, labels, 'business_card_data.json')
    print("💾 數據集已保存")
    
    # 初始化訓練器
    trainer = RobertaNERTrainer()
    
    # 準備數據
    train_loader, val_loader = trainer.prepare_data(texts, labels, batch_size=16)
    
    # 訓練模型
    trainer.train(train_loader, val_loader, epochs=20, learning_rate=2e-5)
    
    print("\n🎉 訓練完成！")
    print("📁 最佳模型已保存為: best_ner_model.pth")
    print("\n📝 下一步:")
    print("1. 運行 'python predict.py' 測試基礎預測功能")
    print("2. 運行 'python predict_roberta.py' 測試RoBERTa預測功能")
    print("3. 運行 'python improved_predict.py' 測試改進版本")
    print("4. 運行 'python predict.py --interactive' 進入互動模式")

if __name__ == "__main__":
    main() 