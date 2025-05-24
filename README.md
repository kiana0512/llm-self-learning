# 🌍 English-to-Chinese Machine Translation with DeepSeek-LLM + LoRA

This project fine-tunes the open-source DeepSeek-LLM on the [WMT19 Zh-En dataset](https://www.statmt.org/wmt19/translation-task.html) using parameter-efficient LoRA training. The pipeline includes data preprocessing, tokenization, training, inference, evaluation, deployment, and a web UI.

---

## 📌 Project Structure

```bash
Fine-tuning/
├── data/                 # Raw + tokenized data
├── preprocess/           # Scripts for data preprocessing
├── train/                # Training with LoRA
├── inference/            # Inference scripts + API + Web UI
├── deployment/           # Final merged model for deployment
├── requirements.txt
├── README.md / README_zh.md
```

---

## 🧪 Features

- ✅ Hugging Face Transformers compatible
- ✅ Efficient fine-tuning with LoRA
- ✅ Support for GPU/CPU inference
- ✅ BLEU evaluation and batched result export
- ✅ Flask API service and lightweight Web UI

---

## 🧱 Tech Stack & Dependencies

| Library        | Description                      |
|----------------|----------------------------------|
| `transformers` | Hugging Face model framework     |
| `peft`         | Parameter-efficient fine-tuning  |
| `datasets`     | Data loading and formatting      |
| `bitsandbytes` | 4-bit quantization support       |
| `evaluate`     | BLEU scoring                     |
| `flask`        | RESTful API + Web interface      |
| `torch`        | Deep learning backend            |

Install via:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Description

- Source: WMT19 En-Zh Translation Task
- Format: JSONL with `prompt` and `response`

Example:

```json
{"prompt": "Translate to Chinese: How are you?", "response": "你好吗？"}
```

---

## 🚀 Quick Start

```bash
# Preprocess
python preprocess/preprocess_wmt19.py
python preprocess/tokenize_and_prepare.py

# Train
python train/train_lora.py

# Inference (single sample)
python inference/infer_one.py

# Batch generation
python inference/generate_predictions.py

# BLEU evaluation
python inference/evaluate_bleu.py

# Launch API
python inference/app_api.py

# Launch Web UI
python inference/app_webui.py
```

---

## 🧊 Model Merge & Deployment

Merge adapter and save model for deployment:

```bash
python save_merged_model.py
```

Saves to: `deployment/merged_model` and `.zip`

---

## 📊 Example Output

Source:

```
The challenge will be to reach an agreement that guarantees the wellbeing of future EU-UK relations.
```

Translation:

```
挑战在于达成一份保障未来欧盟-英国关系福祉的协议。
```

---

## 📬 Acknowledgments

Built with ChatGPT, DeepSeek-LLM, Hugging Face Transformers, and WMT19. Thanks to the open-source community!
