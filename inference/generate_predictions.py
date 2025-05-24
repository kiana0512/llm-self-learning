import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm

def load_model(
    base_model="deepseek-ai/deepseek-llm-7b-base",
    adapter_path="../train/lora_output"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
        local_files_only=True
    )
    return tokenizer, model

def translate_en2zh_batch(dataset_path, output_path):
    tokenizer, model = load_model()
    results = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Translating"):
        item = json.loads(line)
        src = item["prompt"].replace("请将以下英文翻译成中文：", "").strip()
        prompt = f"请将以下英文翻译成中文：{src}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.7
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append(output_text.strip())

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(results))

    print(f"✅ 完成翻译，共 {len(results)} 条，保存至 {output_path}")

if __name__ == "__main__":
    dataset_path = "../data/wmt19_zh_en.jsonl"
    output_path = "../data/preds.txt"
    translate_en2zh_batch(dataset_path, output_path)
