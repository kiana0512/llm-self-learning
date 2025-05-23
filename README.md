# 📘 Fine-tuning DeepSeek-LLM for Chinese-English Translation

## 🔍 Overview

This project fine-tunes the DeepSeek-LLM 7B model on the WMT19 Chinese-English dataset using Hugging Face Transformers. With PEFT (LoRA) and bitsandbytes 4-bit quantization, we enable memory-efficient fine-tuning of large language models for translation tasks.

## 🧠 Objectives

- Load and preprocess WMT19 zh-en data
- Use DeepSeek-LLM 7B as the base model
- Apply LoRA for parameter-efficient tuning
- Use 4-bit quantization for memory efficiency
- Enable downstream inference and evaluation

## 🗂️ Project Structure

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

## 📦 Tech Stack

| Library            | Description                           |
|--------------------|---------------------------------------|
| `transformers` ≥ 4.40 | HF LLM training and loading           |
| `peft` ≥ 0.10          | Efficient fine-tuning with LoRA       |
| `datasets`            | Dataset preprocessing and serialization |
| `bitsandbytes`        | 4-bit quantization backend             |
| `torch`               | Deep learning framework               |
| `accelerate`          | Device and training accelerator       |

## 📚 Dataset

- Source: [WMT19 zh-en translation task](https://www.statmt.org/wmt19/translation-task.html)
- Format: raw parallel corpus (Chinese → English)
- Tokenization: DeepSeek tokenizer, stored as HF Dataset format

## 🚀 Fine-tuning Procedure

```bash
conda activate deepseek_lora
python train/train_lora.py
```

Configuration:

- Epochs: 3
- Batch Size: 2
- LoRA: r=8, alpha=16, dropout=0.05
- Quantization: 4bit + bf16

## 📈 Output and Inference

Model is saved to:

```bash
./lora_output/
```

Load with:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./lora_output")
```

## 🔧 Environment Setup

```bash
conda create -n deepseek_lora python=3.10 -y
conda activate deepseek_lora

pip install torch==2.2.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.1 peft==0.10.0 datasets accelerate bitsandbytes safetensors tokenizers tqdm
```

## 📌 Author

- GitHub: [kiana0512](https://github.com/kiana0512)
- Project Path: `F:\python_project\Fine-tuning`
