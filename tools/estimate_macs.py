from functools import partial
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# https://www.deepspeed.ai/tutorials/flops-profiler/#example-bert

######################### create my model, you should replace all modules with their slow version
bf16 = True

config = AutoConfig.from_pretrained("../model_libs/configs/1B-ffnMoE-1024", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("../llama2_tokenizer")

config.vocab_size += 1
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config.pad_token_id = tokenizer.pad_token_id

if bf16:
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

#########################
def input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.eos_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    inputs = dict(inputs)
    # labels = torch.tensor([1] * batch_size)
    # inputs.update({"labels": labels})
    return inputs


with get_accelerator().device(7):
    batch_size = 4
    seq_len = 1024
    enable_profile = True
    if enable_profile:
      flops, macs, params = get_model_profile(
          model,
          kwargs=input_constructor(batch_size, seq_len, tokenizer),
          print_profile=True,
          detailed=True,
          warm_up=0,
      )
    else:
      inputs = input_constructor((batch_size, seq_len), tokenizer)
      outputs = model(inputs)