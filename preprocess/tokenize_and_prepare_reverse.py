from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from pathlib import Path
import os

# 模型和路径
model_name = "deepseek-ai/deepseek-llm-7b-base"
data_path = "../data/wmt19_en_zh_reversed.jsonl"
save_path = "../data/tokenized_wmt19_en_zh_reversed"

# 加载数据集
print("📂 加载数据文件:", data_path)
raw_dataset = Dataset.from_json(data_path)

# 加载分词器
print("🔄 加载 Tokenizer：", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 分词函数
def prepare_dataset(example):
    model_input = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["response"], truncation=True, padding="max_length", max_length=512)
    model_input["labels"] = labels["input_ids"]
    return model_input

# 执行映射
print("🧹 执行分词...")
tokenized_dataset = raw_dataset.map(prepare_dataset)

# 保存到磁盘
print(f"💾 保存至 {save_path}")
tokenized_dataset.save_to_disk(save_path)
print(f"✅ 数据集构建完成。样本总数: {len(tokenized_dataset)}")
