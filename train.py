# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import argparse
import multiprocessing
import os

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="a_run")
    parser.add_argument("--config_name", type=str, default="required")
    parser.add_argument("--dataset_text_field", type=str, default="content")
    parser.add_argument('--deepspeed', type=str, default="./deespeed_config.json")

    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--global_batch_size", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--learning_rate", type=float, default=4.2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--wandb_project", type=str, default="new-hf-attention-moe")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_position_embeddings", type=int, default=-1)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--disable_wandb", action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--dataset_100bt', action='store_true', default=False)
    parser.add_argument('--wikitext_103', action='store_true', default=False)
    parser.add_argument('--ignore_data_skip', action='store_true', default=False)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--disable_iter_shuffle', action='store_true', default=False, help="whether disable shuffle to iter dataset")
    parser.add_argument('--ensure_pad_token', action='store_true', default=False, help="Code will ensure there is an independent pad token in tokenizer.")

    return parser.parse_args()


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


def main(args):
    assert args.global_batch_size == args.world_size * args.micro_batch_size * args.gradient_accumulation_steps
    
    # os.environ["HF_DATASETS_CACHE"]=""
    # os.environ['HF_HOME'] = ''
    
    if args.disable_wandb:
        report_to = "none"
    else:
        report_to = "wandb"

    wandb_project = args.wandb_project
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        # os.environ["WANDB_API_KEY"] = ""

    # load config
    config = AutoConfig.from_pretrained(os.path.join("model_libs/configs", args.config_name), trust_remote_code=True)
    if args.max_position_embeddings > 0:
        config.max_position_embeddings = args.max_position_embeddings
        print("Max seq len: {}".format(config.max_position_embeddings))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llama2_tokenizer/")

    if args.ensure_pad_token:
        if tokenizer.pad_token_id is None:
            assert config.pad_token_id is None

            print("ADD pad token!\n" + "*"*100)
            config.vocab_size += 1
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            config.pad_token_id = tokenizer.pad_token_id
        else:
            assert config.pad_token_id == tokenizer.pad_token_id
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    assert len(tokenizer) == config.vocab_size

    if args.bf16:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print(model)
    print_trainable_parameters(model)


    # load dataset
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    if args.wikitext_103:
        train_data = load_dataset("dlwh/wikitext_103_detokenized", split='train', num_proc=16, cache_dir=cache_dir)
        train_data = train_data.shuffle()
        train_data = train_data.to_iterable_dataset(num_shards=64)
        eval_data = load_dataset("dlwh/wikitext_103_detokenized", split='validation', cache_dir=cache_dir)
    else:
        # we download fineweb-edu from https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT in advance
        data_files = {'train': 'fineweb_edu_100bt/sample/100BT/*.parquet'}
        train_data = load_dataset('parquet', data_files=data_files, split='train', num_proc=16, cache_dir=cache_dir)
        train_data = train_data.to_iterable_dataset(num_shards=64)
        eval_data = None

        if not args.disable_iter_shuffle:
            raise Exception("I recommend disabling shuffle now. If shuffle is enabled, strange loss curves will appear - there might be a peak. If you insist on using it, please comment out this line.")
            train_data = train_data.shuffle(seed=args.seed, buffer_size=args.global_batch_size * 64)
        else:
            print("Shuffle for Iter dataset is disabled. (Not important.)\n" + "*"*100)

    ################## for eval loss
    class MySFTTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only: bool,
            ignore_keys,):
            
            inputs = self._prepare_inputs(inputs)
            inputs['calculate_loss_without_label'] = True

            assert self.label_smoother is None

            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            return (loss, None, None)
    #############################################

    # setup the trainer
    trainer = MySFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            output_dir=os.path.join(args.output_dir, args.run_name),
            optim="adamw_torch",
            adam_beta2=0.95,
            seed=args.seed,
            run_name=f"{args.run_name}",
            report_to=report_to,
            packing=True,
            eval_packing=False,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            #########
            eval_strategy="steps" if args.wikitext_103 else "noe",
            metric_for_best_model="loss",
            eval_steps=args.save_steps,
            eval_on_start=True if args.wikitext_103 else False,
            load_best_model_at_end=True if args.wikitext_103 else False,
            #########
            dataset_text_field=args.dataset_text_field,
            max_seq_length=config.max_position_embeddings,
            dataset_num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
            deepspeed=args.deepspeed,
            ignore_data_skip=args.ignore_data_skip,
            gradient_checkpointing=args.gradient_checkpointing,
        ),
    )

    # launch
    print("Training...")
    trainer.train(args.resume)

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, args.run_name, "final_checkpoint/"))
    if args.push_to_hub:
        trainer.push_to_hub("Upload model")
    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)

    logging.set_verbosity_error()

    main(args)
