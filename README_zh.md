# 🌍 英中机器翻译模型微调项目（基于 DeepSeek）

本项目基于 [WMT19 中文-英文数据集](https://www.statmt.org/wmt19/translation-task.html) 对开源大语言模型 DeepSeek-LLM 进行参数高效微调（LoRA），实现了高质量的英文到中文翻译。项目流程涵盖数据处理、分词、训练、推理、评估、服务部署及 Web UI 接入。

---

## 📌 项目结构

```bash
Fine-tuning/
├── data/                 # 数据集与预处理
│   ├── wmt19_zh_en.jsonl
│   └── tokenized_wmt19_zh_en/
├── preprocess/           # 数据预处理脚本
├── train/                # LoRA 微调训练脚本与结果
│   └── lora_output/
├── inference/            # 推理脚本、API 服务与 Web UI
├── deployment/           # 合并保存后的模型（部署版本）
├── requirements.txt
├── README.md / README_zh.md
```

---

## 🧪 功能特性

- ✅ 支持 Hugging Face Transformers 大模型框架
- ✅ 使用 LoRA 微调技术降低训练显存成本
- ✅ 支持 GPU/CPU 推理与多轮生成
- ✅ BLEU 分数评估与翻译结果批量导出
- ✅ Flask 接口服务部署 + Web UI 可视化界面

---

## 🧱 技术栈与依赖

| 库 | 说明 |
|----|------|
| `transformers` | Hugging Face 模型加载框架 |
| `peft`         | LoRA 微调模块 |
| `datasets`     | 预处理与加载数据 |
| `bitsandbytes` | 模型量化 (4bit) 支持 |
| `evaluate`     | BLEU 评估指标计算 |
| `flask`        | 提供 RESTful API 与 Web 服务 |
| `torch`        | 深度学习训练框架 |

安装方式：

```bash
pip install -r requirements.txt
```

---

## 📂 数据说明

- 来源：WMT19 英中翻译任务数据
- 格式：每行一个 JSON，包含 `prompt` 与 `response`

示例：

```json
{"prompt": "请将以下英文翻译成中文：How are you?", "response": "你好吗？"}
```

---

## 🚀 快速使用

```bash
# 数据预处理
python preprocess/preprocess_wmt19.py
python preprocess/tokenize_and_prepare.py

# 训练模型（LoRA）
python train/train_lora.py

# 单条推理测试
python inference/infer_one.py

# 批量生成翻译结果
python inference/generate_predictions.py

# BLEU 评估
python inference/evaluate_bleu.py

# 启动 API 服务
python inference/app_api.py

# 启动 Web UI
python inference/app_webui.py
```

---

## 🧊 模型合并部署

运行脚本将 LoRA 微调权重合并保存：

```bash
python save_merged_model.py
```

保存位置：`deployment/merged_model` 和 `.zip` 版本

---

## 📊 示例翻译结果

原文：

```
The challenge will be to reach an agreement that guarantees the wellbeing of future EU-UK relations.
```

微调后翻译：

```
挑战在于达成一份保障未来欧盟-英国关系福祉的协议。
```

---

## 📬 联系与致谢

本项目由 ChatGPT + DeepSeek 模型驱动，感谢 Hugging Face 社区、WMT 数据集提供方的支持。
