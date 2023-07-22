
# S2 a10g num-adapter
python ../../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 13500 --no-mem-pool
python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_no_mem.jsonl

# 20
python ../../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 10000 --no-mem-pool
python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_no_mem.jsonl --append

# 50 100 200
python ../../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 9000 --no-mem-pool
python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_no_mem.jsonl --append




# S2 a10g req rate
python ../../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 9000 --no-mem-pool
python ../../../run_exp.py  --backend slora  --suite a10g-req-rate  --model-setting S2  --mode synthetic  --output synthetic_req_rate_a10g_S2_no_mem.jsonl 
