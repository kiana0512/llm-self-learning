# train/train_lora.py

import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

def main():
    # ✅ 模型与数据路径配置
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    dataset_path = "../data/tokenized_wmt19_zh_en"
    output_dir = "./lora_output"

    # ✅ 加载分词数据集
    print("📂 加载数据集...")
    dataset = load_from_disk(dataset_path)

    # ✅ 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ✅ 设置 4bit 量化配置（不再传 load_in_4bit）
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # ✅ 加载模型（使用量化配置）
    print("📦 加载模型中...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    # ✅ 应用 LoRA 微调模块
    print("🔧 应用 LoRA...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # ✅ 数据集切分（98% 训练 + 2% 验证）
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.02)
    train_data = split["train"]
    eval_data = split["test"]

    # ✅ 训练参数设置
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,  # 若不支持，可改为 fp16=True
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        report_to="none"
    )

    # ✅ 自动填充器：适配语言模型训练需求
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    # ✅ 构建 Trainer
    print("🚀 启动训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("✅ 训练完成，模型保存在：", output_dir)

if __name__ == "__main__":
    main()
