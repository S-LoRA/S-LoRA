## a100-80 S2
# bmm
# python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 200  --num-token 125000 --bmm

# num adapters = 20
# python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 200  --num-token 70000 --bmm

python ../../../run_exp.py  --backend slora  --suite a100-80-S2-num-adapter-bmm  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_bmm.jsonl --append
