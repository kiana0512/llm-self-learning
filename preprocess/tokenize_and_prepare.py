# preprocess/tokenize_and_prepare.py

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_jsonl(filepath):
    """读取 .jsonl 文件，返回一个 list[dict]"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def tokenize_sample(example, tokenizer, max_length=512):
    """
    将单个样本中的 prompt + response 拼接为输入，进行分词。

    注意：
    - input_ids 和 labels 一样，都是 prompt + response 的 token。
    - 如果希望只训练 response，可做额外 masking（高级策略）。
    """
    text = example["prompt"] + "\n\n" + example["response"]

    # tokenizer.encode_plus 返回包含 input_ids、attention_mask 等字段
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # 添加 labels 字段（语言模型的监督目标）
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def prepare_dataset(jsonl_path, tokenizer_name, max_length=512):
    """
    加载 jsonl 数据 → 分词 → 构建 Dataset

    参数说明：
    - jsonl_path: 第一步保存的数据文件路径
    - tokenizer_name: 模型的 tokenizer 名称（如 deepseek-ai/deepseek-llm-7b-base）
    """
    print(f"📂 加载数据文件: {jsonl_path}")
    raw_data = load_jsonl(jsonl_path)

    print(f"🔄 加载 Tokenizer：{tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print("🧹 执行分词...")
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(lambda x: tokenize_sample(x, tokenizer, max_length), remove_columns=["prompt", "response"])

    print("✅ 数据集构建完成。样本总数:", len(dataset))
    return dataset, tokenizer

if __name__ == "__main__":
    # 示例调用
    dataset,tokenizer = prepare_dataset(
        jsonl_path="../data/wmt19_zh_en.jsonl",
        tokenizer_name="deepseek-ai/deepseek-llm-7b-base",  # 可根据实际修改
        max_length=512
    )

    # 示例输出首条数据
    print(dataset[0])
    print(tokenizer.decode(dataset[0]["input_ids"], skip_special_tokens=True))

    # ✅ 保存成磁盘数据集以供训练复用
    dataset.save_to_disk("../data/tokenized_wmt19_zh_en")
    print("✅ 分词数据集已保存至 ../data/tokenized_wmt19_zh_en")
