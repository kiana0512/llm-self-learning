import json
from pathlib import Path

# 原始数据集路径
input_path = Path("../data/wmt19_zh_en.jsonl")  # 修改为你实际路径
output_path = Path("../data/wmt19_en_zh_reversed.jsonl")  # 新输出路径

# 执行翻转：prompt 和 response 对调，并修改提示语
with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line.strip())
        zh = item["response"].strip()
        en = item["prompt"].replace("请将以下英文翻译成中文：", "").strip()
        new_item = {
            "prompt": f"请将以下中文翻译成英文：{zh}",
            "response": en
        }
        fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"✅ 已生成翻转数据集文件: {output_path}")
