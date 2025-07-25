命令是
```bash
export TOKENIZERS_PARALLELISM=false
python preprocess.py --seq_len 1024 --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --split train
```