## a10g S2
# lifo
# python launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 14500
python ../../../run_exp.py  --backend slora  --suite a10g-cv  --model-setting S2  --mode synthetic  --output synthetic_cv_a10g_S2_lifo.jsonl

## a100-80g S4
# lifo
# python launch_server.py  --device a100  --model-setting S4  --backend slora  --num-adapter 400  --num-token 65000
python python ../../../run_exp.py  --backend slora  --suite a100-80-S4-cv  --model-setting S4  --mode synthetic  --output synthetic_cv_a100-80_S4_lifo.jsonl