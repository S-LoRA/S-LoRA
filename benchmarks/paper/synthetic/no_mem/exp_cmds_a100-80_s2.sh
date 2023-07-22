## a100-80 S2
# slora-no-mem
python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 500  --num-token 125000 --no-mem
python ../../../run_exp.py  --backend slora  --suite a100-80-S2-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_no_mem.jsonl &&

# num adapter = 50
python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 500  --num-token 110000 --no-mem
python ../../../run_exp.py  --backend slora  --suite a100-80-S2-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_no_mem.jsonl --append

# num adapter = 100
python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 500  --num-token 100000 --no-mem
python ../../../run_exp.py  --backend slora  --suite a100-80-S2-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_no_mem.jsonl --append

# num adapter = 200
python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 500  --num-token 90000 --no-mem
python ../../../run_exp.py  --backend slora  --suite a100-80-S2-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_no_mem.jsonl --append
