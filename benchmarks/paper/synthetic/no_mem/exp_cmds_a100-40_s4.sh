## a100-80 S4
# slora-no-mem
# python launch_server.py  --device a100-40  --model-setting S4  --backend slora  --num-adapter 200  --num-token 14000 --no-mem
# num adapter = 20
# python launch_server.py  --device a100-40  --model-setting S4  --backend slora  --num-adapter 200  --num-token 9000 --no-mem
python ../../../run_exp.py  --backend slora  --suite a100-40-num-adapter  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100-40_S4_no_mem.jsonl --append

