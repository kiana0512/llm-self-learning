import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(base_model="deepseek-ai/deepseek-llm-7b-base", adapter_path="./lora_output"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16)
    adapter_path = os.path.abspath("../train/lora_output")  # 获取绝对路径
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
        local_files_only=True
    )
    return tokenizer, model

def translate(text, tokenizer, model, max_new_tokens=100):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model = load_model()

    test_text = "请将以下英文翻译成中文：The challenge will be to reach an agreement that guarantees the wellbeing of future EU-UK relations."
    result = translate(test_text, tokenizer, model)
    print("原文:", test_text)
    print("翻译:", result)
