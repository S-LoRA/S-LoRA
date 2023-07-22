## a100-40 S2
# bmm
# python launch_server.py  --device a100-40  --model-setting S4  --backend slora  --num-adapter 200  --num-token 14000 --bmm
# python ../../../run_exp.py  --backend slora  --suite a100-40-num-adapter-short  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100-40_S4_bmm.jsonl --append

# num adapters = 20
# python launch_server.py  --device a100-40  --model-setting S4  --backend slora  --num-adapter 200  --num-token 7000 --bmm
python ../../../run_exp.py  --backend slora  --suite a100-40-num-adapter-short  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100-40_S4_bmm.jsonl --append
