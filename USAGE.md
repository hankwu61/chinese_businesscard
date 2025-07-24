# 中文名片NER项目使用说明

## 项目概述

这是一个基于BERT的中文名片命名实体识别（NER）项目，可以自动提取名片中的各种信息，包括姓名、职衔、电话、手机、公司、邮箱、地址、统一编号和网站等。

## 快速开始

### 1. 环境准备

确保已安装所需的依赖包：

```bash
pip install -r requirements.txt
```

### 2. 测试环境

运行测试脚本验证环境是否正确配置：

```bash
python test_model.py
```

### 3. 查看演示

运行演示脚本了解项目功能：

```bash
python demo.py
```

## 主要功能

### 数据生成

项目包含一个数据处理器，可以自动生成中文名片训练数据：

```python
from data_processor import BusinessCardDataProcessor

processor = BusinessCardDataProcessor()
texts, labels = processor.generate_dataset(num_samples=1000)
```

### 模型训练

使用以下命令开始训练模型：

```bash
python train.py
```

或者使用快速训练脚本：

```bash
python quick_train.py
```

### 模型预测

训练完成后，可以使用预测脚本提取名片信息：

```bash
# 使用示例文本
python predict.py

# 使用自定义文本
python predict.py --text "張三 總經理 02-1234-5678 0912345678 依林一集團"

# 交互模式
python predict.py --interactive
```

## 项目结构

```
chinese_businesscard/
├── data_processor.py      # 数据处理和生成模块
├── chinese_ner_model.py   # 模型定义和数据集类
├── train.py              # 模型训练脚本
├── predict.py            # 模型预测脚本
├── test_model.py         # 环境测试脚本
├── demo.py               # 功能演示脚本
├── requirements.txt      # 依赖包列表
└── README.md            # 项目说明
```

## 支持的实体类型

项目支持以下10种实体类型的识别：

1. **姓名** (NAME) - 人名
2. **职衔** (TITLE) - 职位头衔
3. **电话** (PHONE) - 固定电话
4. **传真** (FAX) - 传真号码
5. **手机** (MOBILE) - 手机号码
6. **公司** (COMPANY) - 公司名称
7. **邮箱** (EMAIL) - 电子邮箱
8. **地址** (ADDRESS) - 地址信息
9. **统一编号** (TAX_ID) - 统一编号
10. **网站** (WEBSITE) - 网站地址

## 使用示例

### 基本使用

```python
from predict import BusinessCardNER

# 初始化预测器
ner = BusinessCardNER()

# 预测名片信息
text = "張三 總經理 02-1234-5678 0912345678 依林一集團 john@example.com"
result = ner.predict_and_format(text)
```

### 自定义训练

```python
from train import NERTrainer
from data_processor import BusinessCardDataProcessor

# 生成训练数据
processor = BusinessCardDataProcessor()
texts, labels = processor.generate_dataset(1000)

# 训练模型
trainer = NERTrainer()
train_loader, val_loader = trainer.prepare_data(texts, labels)
trainer.train(train_loader, val_loader)
```

## 注意事项

1. **模型文件**：首次运行预测时，如果没有训练好的模型文件 `best_ner_model.pth`，会使用未训练的模型，预测结果可能不准确。

2. **GPU支持**：项目支持GPU加速，如果检测到CUDA设备会自动使用GPU。

3. **中文支持**：项目专门针对中文名片优化，使用中文BERT模型。

4. **数据格式**：输入文本应该是空格分隔的名片信息，例如："姓名 职衔 电话 手机 公司 邮箱 地址"。

## 故障排除

### 常见问题

1. **模块导入错误**：确保所有依赖包已正确安装
2. **CUDA错误**：检查GPU驱动和CUDA版本
3. **内存不足**：减少batch_size或使用CPU模式

### 获取帮助

如果遇到问题，请检查：
- Python版本（建议3.8+）
- PyTorch版本兼容性
- 依赖包版本
- 系统内存和GPU内存

## 扩展功能

项目支持以下扩展：

1. **添加新实体类型**：在 `BusinessCardDataset` 中更新标签映射
2. **使用不同BERT模型**：修改 `model_name` 参数
3. **自定义数据**：使用自己的训练数据
4. **模型优化**：调整超参数和训练策略 