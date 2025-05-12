import evaluate
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch


perplexity = evaluate.load("perplexity_metric.py", module_type="metric")

bf16 = True

###################################### load local pretrained model
model_id = 'checkpoint-16500'

if bf16:
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print(model)
# exit()
######################################

from datasets import load_dataset
import numpy as np

data_files = {'train': 'fineweb_100bt_00099.jsonl.gz'} # this file is not used for training
dataset = load_dataset('json', data_files=data_files, split='train', streaming=True)
# dataset = load_dataset("dlwh/wikitext_103_detokenized", split='test')
batched_dataset = dataset.batch(batch_size=64 * 32 * 4)

count = 0.0
sum_ppl = 0.0

step_num = 10
for idx, batch in enumerate(batched_dataset):
    if idx == step_num:
        print(f"avg ppl {round(sum_ppl / count, 3)} of {count} samples!")
        break

    print(f"idx {idx + 1} / {step_num}")

    input_texts = batch['text']
    new_texts = [s for s in input_texts if len(s) > 0]
    results = perplexity.compute(model_id=None,
                                add_start_token=True,
                                predictions=new_texts,
                                bf16=True,
                                max_length=1024,
                                batch_size=64,
                                model=model,
                                tokenizer=tokenizer,)
    
    count += len(results['perplexities'])
    sum_ppl += np.sum(results['perplexities'])
    print(f"avg ppl {round(sum_ppl / count, 3)} of {count} samples!")