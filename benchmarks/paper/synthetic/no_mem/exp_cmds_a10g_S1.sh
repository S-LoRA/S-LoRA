# S1 a10g
# python launch_server.py  --device a10g  --model-setting S1  --backend dm  --num-adapter 200  --num-token 13500 --no-mem-pool
python ../../../run_exp.py  --backend dm  --suite a10g-num-adapter  --model-setting S1  --mode synthetic  --output synthetic_num_adapters_a10g_S1_no_mem.jsonl

# 20
# python ../../../launch_server.py  --device a10g  --model-setting S2  --backend dm  --num-adapter 200  --num-token 10000 --no-mem-pool
# python ../../../run_exp.py  --backend dm  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_no_mem.jsonl --append

# 50
# python ../../../launch_server.py  --device a10g  --model-setting S2  --backend dm  --num-adapter 200  --num-token 9000 --no-mem-pool
# python ../../../run_exp.py  --backend dm  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_no_mem.jsonl --append
