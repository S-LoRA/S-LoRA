# S2 num adapter 1
python ../../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 13000 --bmm --no-mem-pool
python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S2  --mode synthetic --output synthetic_num_adapters_a10g_S2_bmm.jsonl

# 20 50 100 200
python ../../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 8000 --bmm --no-mem-pool
python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S2  --mode synthetic --output synthetic_num_adapters_a10g_S2_bmm.jsonl --append

# S2 req rate
python ../../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 8000 --bmm --no-mem-pool
python ../../../run_exp.py  --backend slora  --suite a10g-req-rate  --model-setting S2  --mode synthetic --output synthetic_req_rate_a10g_S2_bmm.jsonl
