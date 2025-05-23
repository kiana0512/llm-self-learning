import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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

def translate_en2zh(text: str, tokenizer, model, max_new_tokens=150) -> str:
    prompt = f"请将以下英文翻译成中文：{text.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model = load_model()
    input_text = "Fine-tuning of large language models is a meaningful thing."
    output = translate_en2zh(input_text, tokenizer, model)
    print("原文:", input_text)
    print("翻译:", output)
