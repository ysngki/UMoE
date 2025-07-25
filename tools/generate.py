from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import torch

# https://huggingface.co/docs/transformers/en/llm_tutorial

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

# config
# config_name = "small-vanilla-ffnMoE-1024"
# config_name = "1B-ffnMoE-1024"
# config_name = "small-vanilla-UMoE-1024"
# config_name = "small-mla-UMoE-1024"
config_name = "1B-GPT-1024"

config = AutoConfig.from_pretrained(os.path.join("../model_libs/configs", config_name), trust_remote_code=True)
config.gpt_premix = True
# config.gpt_postmix = True

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("../llama2_tokenizer/")

if tokenizer.pad_token_id is None:
    assert config.pad_token_id is None

    print("ADD pad token!\n" + "*"*100)
    config.vocab_size += 1
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.pad_token_id = tokenizer.pad_token_id
else:
    assert config.pad_token_id == tokenizer.pad_token_id

# model
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")

print(model)
print_trainable_parameters(model)


# ###########
print("\n\n")
print("========== SAMPLE GENERATION ==============")

input_ids = torch.randint(0, model.config.vocab_size, (1, 8192)).to("cuda")
while True:
    output = model.generate(input_ids, max_new_tokens=1, return_dict_in_generate=True)

    output_ids = output.sequences
    kv_cache = output.past_key_values
    
    print("cuda memory", torch.cuda.memory_allocated(device='cuda'))

    ####### probe kv cache
    print("\n\n")
    print(f"kv cache:\nLayer Num {len(kv_cache.key_cache)}, {len(kv_cache.value_cache)}\nKey:{kv_cache.key_cache[0].shape}\nValue:{kv_cache.value_cache[0].shape}")
    whole_k = torch.stack(kv_cache.key_cache)
    whole_v = torch.stack(kv_cache.value_cache)
    print(f"K Cache's Byte: {whole_k.element_size() * whole_k.nelement()}")
    print(f"V Cache's Byte: {whole_v.element_size() * whole_v.nelement()}")
    kv_byte_num = whole_v.element_size() * whole_v.nelement() + whole_k.element_size() * whole_k.nelement()
    print(f"KV Cache's Byte: {kv_byte_num} B, {kv_byte_num/(1024*1024)} MB, {kv_byte_num/(1024*1024*1024)} GB")
    raise Exception("!!!!")
    # torch.save(torch.stack([whole_k.view(-1), whole_v.view(-1)]), "temp.pt")
    ######

    # print(tokenizer.decode(output_ids[0]))
print("==========================================\n\n")


# ###########
print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
raise Exception(f"{input_ids}")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# ###########
# # tokenize input
# model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
# # generate
# generated_ids = model.generate(**model_inputs, max_new_tokens=40)
# # tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:]))
# raise Exception("fgenerated_ids:\n{generated_ids}")
