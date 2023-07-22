## a10g s1
# bmm
# python launch_server.py  --device a10g  --model-setting S1  --backend dm  --num-adapter 200  --num-token 14500 --bmm
# python ../../../run_exp.py  --backend dm  --suite a10g-num-adapter  --model-setting S1  --mode synthetic  --output synthetic_num_adapters_a10g_S1_bmm.jsonl 

# num adapters = 20
# python launch_server.py  --device a10g  --model-setting S1  --backend dm  --num-adapter 200  --num-token 12000 --bmm
python ../../../run_exp.py  --backend dm  --suite a10g-num-adapter  --model-setting S1  --mode synthetic  --output synthetic_num_adapters_a10g_S1_bmm.jsonl --append 
