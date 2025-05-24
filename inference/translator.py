from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class Translator:
    def __init__(self, base_model="deepseek-ai/deepseek-llm-7b-base", adapter_path="../train/lora_output"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model = PeftModel.from_pretrained(
            base,
            adapter_path,
            is_trainable=False,
            local_files_only=True
        )

    def translate(self, en_text):
        prompt = f"请将以下英文翻译成中文：{en_text.strip()}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text.strip()
