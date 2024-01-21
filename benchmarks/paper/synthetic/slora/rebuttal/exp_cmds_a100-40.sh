## a100-40g S4
# slora
# python launch_server.py  --device a100  --model-setting S4  --backend slora  --num-adapter 200  --num-token 14000 --dummy
python ../../../../run_exp.py  --backend slora  --suite a100-40-num-adapter-rank-2  --model-setting S-13b-r2  --mode synthetic --output synthetic_num_adapters_a100-40_S-13b-r2_slora.jsonl
