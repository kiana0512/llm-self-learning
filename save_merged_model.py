import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Step 1: 配置路径
adapter_path = "./train/lora_output"
output_path = "./deployment/merged_model"
base_model = "deepseek-ai/deepseek-llm-7b-base"

# Step 2: 加载并合并模型
print("📦 正在加载 Base 模型 + LoRA Adapter...")
base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload()

# Step 3: 保存合并后的模型
print(f"💾 保存至：{output_path}")
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.save_pretrained(output_path)

# Step 4: 可选压缩
shutil.make_archive(output_path, 'zip', output_path)
print("✅ 模型保存并压缩完成。")
