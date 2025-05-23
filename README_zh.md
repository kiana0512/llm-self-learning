# 📘 基于 DeepSeek-LLM 的中英翻译模型微调项目

## 🔍 项目简介

本项目基于 Hugging Face Transformers 框架，采用 DeepSeek-LLM 7B 预训练语言模型，对 WMT19 中文-英文翻译数据集进行微调。通过引入 PEFT 中的 LoRA 技术和 bitsandbytes 的 4bit 参数量化方法，实现高效且低资源消耗的大语言模型微调流程，适用于科研、教学或产品原型开发。

## 🧠 项目目标

- 加载并预处理 WMT19 zh-en 数据集
- 使用 DeepSeek-LLM 作为基础模型
- 应用 LoRA（Low-Rank Adaptation）进行参数高效微调
- 引入 4bit 量化减少显存占用
- 支持后续推理部署与评估（BLEU/chrF）

## 🗂️ 项目结构

```bash
Fine-tuning/
├── data/
│   └── tokenized_wmt19_zh_en/
├── preprocess/
│   ├── load_wmt.py
│   └── tokenize_and_prepare.py
├── train/
│   └── train_lora.py
└── README.md
```

## 📦 技术栈与依赖

| 库               | 说明                         |
|------------------|------------------------------|
| `transformers` ≥ 4.40 | 语言模型/微调主框架          |
| `peft` ≥ 0.10         | 参数高效微调模块（LoRA）     |
| `datasets`           | 数据集处理与序列化           |
| `bitsandbytes`       | 权重量化支持（4bit）         |
| `torch`              | 深度学习训练框架             |
| `accelerate`         | 加速器（自动设备映射）       |

## 📚 数据集说明

- 数据来源：[WMT19 中英翻译任务](https://www.statmt.org/wmt19/translation-task.html)
- 格式：原始中英文本对
- 分词方法：使用 DeepSeek tokenizer，保存为 HF Dataset 格式

## 🚀 微调流程

```bash
conda activate deepseek_lora
python train/train_lora.py
```

训练配置参数：

- Epochs: 3
- Batch Size: 2
- LoRA: r=8, alpha=16, dropout=0.05
- 模型加载：4bit 量化 + bf16 精度

## 📈 模型保存与推理

模型保存路径：

```bash
./lora_output/
```

加载方式：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./lora_output")
```

## 🔧 环境配置推荐

```bash
conda create -n deepseek_lora python=3.10 -y
conda activate deepseek_lora

pip install torch==2.2.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.1 peft==0.10.0 datasets accelerate bitsandbytes safetensors tokenizers tqdm
```

## 📌 作者信息

- GitHub: [kiana0512](https://github.com/kiana0512)
- 项目路径：`F:\python_project\Fine-tuning`
