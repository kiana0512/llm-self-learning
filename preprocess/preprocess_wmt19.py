# preprocess/preprocess_wmt19.py

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
from datasets import load_dataset


def convert_to_instruction(example):
    """将每个样本转换为指令微调格式"""
    en = example["translation"]["en"]
    zh = example["translation"]["zh"]
    return {
        "prompt": f"请将以下英文翻译成中文：{en}",
        "response": zh
    }


def preprocess_wmt19_zh_en(output_path: str, num_samples: int = None):
    """
    加载 WMT19 zh-en 数据集，转换为指令格式，并保存为 jsonl 文件。

    参数：
        output_path (str): 输出文件路径
        num_samples (int): 限制处理样本数量（调试时可设置较小值）
    """
    print("🔄 加载 WMT19 zh-en 数据集...")
    dataset = load_dataset("wmt19", "zh-en", split="train")

    if num_samples:
        dataset = dataset.select(range(num_samples))

    print("🚧 正在转换为指令格式...")
    processed = [convert_to_instruction(item) for item in dataset]

    print(f"💾 正在保存至 {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 成功处理并保存 {len(processed)} 条样本。")


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    preprocess_wmt19_zh_en(output_path="../data/wmt19_zh_en.jsonl", num_samples=10000)  # 可调试用10000
