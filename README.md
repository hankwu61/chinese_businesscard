# 中文名片NER識別系統

這是一個基於HuggingFace Transformers中文BERT模型的名片信息命名實體識別(NER)系統，能夠識別中文和英文名片中的姓名、職稱、公司名稱和電話號碼。

## 功能特點

- 🎯 **多語言支持**: 支持中文和英文名片識別
- 🤖 **BERT模型**: 使用預訓練的中文BERT模型
- 📊 **實體類型**: 識別姓名、職稱、公司、電話四種實體
- 🔄 **自動數據生成**: 內建數據生成器，無需手動標註
- 📈 **完整訓練流程**: 包含訓練、驗證、測試全流程
- 🎮 **互動模式**: 支持命令行互動測試

## 實體類型

| 實體類型 | 描述 | 示例 |
|---------|------|------|
| NAME | 姓名 | 張三、John Smith |
| TITLE | 職稱 | 總經理、CEO、工程師 |
| PHONE | 電話號碼 | 02-1234-5678、555-123-4567 |
| FAX | 傳真號碼 | 02-1234-5678、555-123-4567 |
| MOBILE | 行動電話 | 0912345678、13812345678 |
| COMPANY | 公司名稱 | 阿里巴巴集團、Microsoft |
| EMAIL | 電子郵件 | john@example.com |
| ADDRESS | 地址 | 台北市信義區信義路五段7號 |
| TAX_ID | 統一編號 | 12345678、123456789 |
| WEBSITE | 網站 | www.example.com、https://company.com |

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 快速開始

### 1. 測試模型環境

```bash
python test_model.py
```

### 2. 訓練模型

```bash
python train.py
```

訓練過程會：
- 自動生成2000個訓練樣本
- 使用80%數據訓練，20%數據驗證
- 保存最佳模型到 `best_ner_model.pth`

### 3. 測試模型

```bash
# 運行預設測試案例
python predict.py

# 進入互動模式
python predict.py --interactive
```

## 項目結構

```
chinese_businesscard/
├── requirements.txt          # 依賴包列表
├── chinese_ner_model.py      # 核心模型定義
├── data_processor.py         # 數據處理和生成
├── train.py                  # 模型訓練腳本
├── predict.py                # 模型預測腳本
├── test_model.py             # 環境測試腳本
├── README.md                 # 項目說明
├── business_card_data.json   # 生成的訓練數據
└── best_ner_model.pth        # 訓練好的模型
```

## 使用示例

### 訓練示例

```python
from train import NERTrainer
from data_processor import BusinessCardDataProcessor

# 生成數據
processor = BusinessCardDataProcessor()
texts, labels = processor.generate_dataset(1000)

# 訓練模型
trainer = NERTrainer()
train_loader, val_loader = trainer.prepare_data(texts, labels)
trainer.train(train_loader, val_loader)
```

### 預測示例

```python
from predict import BusinessCardNER

# 初始化模型
ner = BusinessCardNER()

# 預測名片信息
text = "張三 總經理 02-1234-5678 0912345678 阿里巴巴集團 john@example.com 台北市信義區信義路五段7號 12345678 www.example.com"
result = ner.predict_and_format(text)

print(f"姓名: {result['summary']['name']}")
print(f"職稱: {result['summary']['title']}")
print(f"電話: {result['summary']['phone']}")
print(f"傳真: {result['summary']['fax']}")
print(f"行動: {result['summary']['mobile']}")
print(f"公司: {result['summary']['company']}")
print(f"電子郵件: {result['summary']['email']}")
print(f"地址: {result['summary']['address']}")
print(f"統一編號: {result['summary']['tax_id']}")
print(f"網站: {result['summary']['website']}")
```

## 模型架構

- **基礎模型**: `hfl/chinese-roberta-wwm-ext`
- **分類層**: 線性層 + Dropout
- **標籤方案**: BIO標註方案
- **損失函數**: CrossEntropyLoss
- **優化器**: AdamW
- **學習率調度**: LinearScheduleWithWarmup

## 性能指標

模型在測試集上的典型性能：
- **Precision**: ~0.95
- **Recall**: ~0.93
- **F1-Score**: ~0.94

## 自定義和擴展

### 添加新的實體類型

1. 在 `BusinessCardDataset` 中更新 `label2id` 映射
2. 在 `BusinessCardDataProcessor` 中添加新的實體生成邏輯
3. 更新模型的分類層輸出維度

### 使用不同的BERT模型

```python
# 使用其他中文BERT模型
model = ChineseNERModel(model_name="hfl/chinese-bert-wwm-ext")
```

### 自定義數據

```python
# 使用自己的數據
texts = ["你的名片文本1", "你的名片文本2"]
labels = [["B-NAME", "I-NAME", "O", ...], ...]

dataset = BusinessCardDataset(texts, labels, tokenizer)
```

## 注意事項

1. **GPU要求**: 建議使用GPU進行訓練，CPU訓練會較慢
2. **內存要求**: 訓練時需要至少4GB內存
3. **數據格式**: 標籤必須使用BIO標註方案
4. **模型大小**: 預訓練模型約400MB，首次運行會自動下載

## 故障排除

### 常見問題

1. **CUDA內存不足**: 減少batch_size或使用CPU訓練
2. **模型下載失敗**: 檢查網絡連接，或手動下載模型
3. **標籤對齊錯誤**: 確保文本和標籤長度一致

### 調試建議

```bash
# 詳細調試信息
python -u train.py 2>&1 | tee training.log

# 檢查模型結構
python -c "from chinese_ner_model import ChineseNERModel; model = ChineseNERModel(); print(model)"
```

## 貢獻

歡迎提交Issue和Pull Request來改進這個項目！

## 許可證

MIT License 