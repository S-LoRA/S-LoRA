## a100-80 S4
# slora-no-mem
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 400  --num-token 65000 --no-mem
# num adapter = 100
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 400  --num-token 40000 --no-mem
# num adapter = 400
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 400  --num-token 35000 --no-mem
# python ../../../run_exp.py  --backend slora  --suite a100-80-S4-num-adapter  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100-80_S4_no_mem.jsonl --append
python ../../../run_exp.py  --backend slora  --suite a100-80-S4-req-rate  --model-setting S4  --mode synthetic  --output synthetic_req_rate_a100-80_S4_no_mem.jsonl

# num adapter = 200
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 500  --num-token 100000 --no-mem
# python ../../../run_exp.py  --backend slora  --suite a100-80-num-adapter  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100-80_S4_no_mem.jsonl --append

# num adapter = 400
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 500  --num-token 90000 --no-mem
# python ../../../run_exp.py  --backend slora  --suite a100-80-num-adapter  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100-80_S4_no_mem.jsonl --append
