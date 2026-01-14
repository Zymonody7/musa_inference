#!/usr/bin/env python3
"""
使用LlamaFactory进行微调的脚本
"""
import os
import json
import subprocess
import sys

# ===== 评测机网络隔离环境配置（必须在 import torch 之前）=====
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_MUSA_ALLOC_CONF"] = "max_split_size_mb:128"
# MUSA设备需要eager attention
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"


# 配置路径
LOCAL_MODEL_PATH = "./local-model"
DATA_PATH = "./data/qa_data.json"  # 使用完整数据集
OUTPUT_DIR = "./finetuned_model"  # 使用正式输出目录

# 检查数据
print("检查数据格式...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"数据总量: {len(data)}")

dataset_dir = "./data/qa_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# 将数据转换为LlamaFactory需要的格式
train_file = os.path.join(dataset_dir, "train.json")
with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"数据集已保存到: {train_file}")

# 创建数据集信息文件（LlamaFactory需要dataset_info.json）
dataset_info = {
    "qa_dataset": {
        "file_name": "qa_dataset/train.json",  # 相对于data目录的路径
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
}
dataset_info_file = "./data/dataset_info.json"
with open(dataset_info_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)
print(f"数据集信息文件已保存到: {dataset_info_file}")

# 构建LlamaFactory命令行
# 使用全量微调（full）以实现过拟合，LoRA不足以实现强过拟合
cmd = [
    "llamafactory-cli", "train",
    "--stage", "sft",
    "--do_train",
    "--model_name_or_path", LOCAL_MODEL_PATH,
    "--dataset", "qa_dataset",
    "--template", "qwen",
    "--cutoff_len", "512",
    "--overwrite_cache",
    "--overwrite_output_dir",
    "--output_dir", OUTPUT_DIR,
    "--finetuning_type", "lora",
    "--lora_rank", "256", 
    "--lora_alpha", "512", 
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "8",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "20",
    "--save_steps", "100",
    "--learning_rate", "5e-5",
    "--num_train_epochs", "40",
    "--max_grad_norm", "0.5",
    "--flash_attn", "disabled",  # MUSA设备需要禁用flash attention
    # MUSA设备不使用fp16和gradient_checkpointing（会导致attention错误）
]

print("开始微调...")
print(f"执行命令: {' '.join(cmd)}")

# 执行训练
result = subprocess.run(cmd, check=False)
sys.exit(result.returncode)
