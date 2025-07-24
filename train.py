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

class NERTrainer:
    """NER模型訓練器"""
    
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
        
    def prepare_data(self, texts: list, labels: list, batch_size: int = 16):
        """準備數據"""
        dataset = BusinessCardDataset(texts, labels, self.tokenizer)
        
        # 分割訓練集和驗證集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs: int = 10, learning_rate: float = 2e-5):
        """訓練模型"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding標籤
        
        # 學習率調度器
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
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
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_ner_model.pth')
                print("保存最佳模型")
    
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
    print("開始訓練中文名片NER模型...")
    
    # 初始化數據處理器
    processor = BusinessCardDataProcessor()
    
    # 生成或加載數據
    data_file = 'business_card_data.json'
    if os.path.exists(data_file):
        print("加載現有數據集...")
        texts, labels = processor.load_dataset(data_file)
    else:
        print("生成新的數據集...")
        texts, labels = processor.generate_dataset(num_samples=2000)
        processor.save_dataset(texts, labels, data_file)
    
    print(f"數據集大小: {len(texts)} 個樣本")
    
    # 初始化訓練器
    trainer = NERTrainer()
    
    # 準備數據
    train_loader, val_loader = trainer.prepare_data(texts, labels, batch_size=16)
    
    # 訓練模型
    trainer.train(train_loader, val_loader, epochs=5, learning_rate=2e-5)
    
    print("訓練完成！")

if __name__ == "__main__":
    main() 