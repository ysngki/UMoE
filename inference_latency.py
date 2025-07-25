import argparse
import multiprocessing
import os
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    logging,
    set_seed,
    AutoTokenizer,
    AutoConfig,
)
from trl import SFTTrainer, SFTConfig


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def test_model_inference(model, config, sequence_length=10, batch_size=1, num_runs=10):
    """
    Test model inference performance with dummy input
    """
    # 设置模型为评估模式
    model.eval()
    
    # 创建dummy input tensor
    # 假设vocab_size可以从config获取，如果没有则使用一个合理的默认值
    vocab_size = getattr(config, 'vocab_size', 50257)  # GPT-2的vocab_size作为默认值
    
    # 生成随机token ids作为输入
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    print(f"Testing inference with input shape: {dummy_input_ids.shape}")
    print(f"Input tensor: {dummy_input_ids}")
    
    # 将输入移到GPU
    device = model.device  # 使用模型所在的设备
    dummy_input_ids = dummy_input_ids.to(device)
    
    print(f"Using device: {device}")
    
    # 预热（第一次推理可能会慢一些）
    with torch.no_grad():
        for i in range(num_runs):
            _ = model(dummy_input_ids)
    
    print(f"Begin!!!")

    # 测试推理时间
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            outputs = model(dummy_input_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            print(f"Run {i+1}: {inference_time:.4f} seconds")

            torch.cuda.empty_cache()
    
    # 计算平均时间
    avg_time = sum(times) / len(times)
    print(f"\nAverage inference time: {avg_time:.4f} seconds")
    print(f"Output shape: {outputs.logits.shape}")
    
    return avg_time, outputs


# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 使用第一张GPU卡
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

# 加载配置和模型
config = AutoConfig.from_pretrained(os.path.join("model_libs/configs", "small-GPT"), trust_remote_code=True)

model = AutoModelForCausalLM.from_config(
    config, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
)

# print(model)
print(model)
print_trainable_parameters(model)
# exit()

# 将模型移到GPU
model = model.to(device)
print(f"Model loaded on device: {model.device}")

# 添加性能测试
print("\n" + "="*50)
print("INFERENCE PERFORMANCE TEST")
print("="*50)

num_runs = 100
batch_size = 64
# 测试不同序列长度
for seq_len in [1024]:
    print(f"\n--- Testing with sequence length: {seq_len} ---")
    avg_time, outputs = test_model_inference(model, config, sequence_length=seq_len, batch_size=batch_size, num_runs=num_runs)