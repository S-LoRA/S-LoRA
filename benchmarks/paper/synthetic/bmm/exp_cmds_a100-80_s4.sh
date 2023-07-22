## a100-80 S4
# bmm
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 400  --num-token 65000 --bmm
# 100
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 400  --num-token 35000 --bmm
# 400
# python launch_server.py  --device a100-80  --model-setting S4  --backend slora  --num-adapter 400  --num-token 35000 --bmm
# python ../../../run_exp.py  --backend slora  --suite a100-80-S4-num-adapter  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100-80_S4_bmm.jsonl --append
python ../../../run_exp.py  --backend slora  --suite a100-80-S4-req-rate  --model-setting S4  --mode synthetic  --output synthetic_req_rate_a100-80_S4_bmm.jsonl

# num adapters = 100

