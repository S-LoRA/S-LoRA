## a10g S2
# abort
# python launch_server.py  --device a10g  --model-setting S2  --backend dm  --num-adapter 200  --num-token 14500 --enable-abort
# python ../../../run_exp.py  --backend dm  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_abort.jsonl
# python ../../../run_exp.py  --backend dm  --suite a10g-alpha  --model-setting S2  --mode synthetic  --output synthetic_alpha_a10g_S2_abort.jsonl
# python ../../../run_exp.py  --backend dm  --suite a10g-cv  --model-setting S2  --mode synthetic  --output synthetic_cv_a10g_S2_abort.jsonl
# python ../../../run_exp.py  --backend dm  --suite a10g-req-rate  --model-setting S2  --mode synthetic  --output synthetic_req_rate_a10g_S2_abort.jsonl

## a10g S1
# abort
# python launch_server.py  --device a10g  --model-setting S1  --backend dm  --num-adapter 200  --num-token 14500 --enable-abort
# python ../../../run_exp.py  --backend dm  --suite a10g-num-adapter  --model-setting S1  --mode synthetic  --output synthetic_num_adapters_a10g_S1_abort.jsonl &&
# python ../../../run_exp.py  --backend dm  --suite a10g-alpha  --model-setting S1  --mode synthetic  --output synthetic_alpha_a10g_S1_abort.jsonl &&
# python ../../../run_exp.py  --backend dm  --suite a10g-cv  --model-setting S1  --mode synthetic  --output synthetic_cv_a10g_S1_abort.jsonl &&
# python ../../../run_exp.py  --backend dm  --suite a10g-req-rate  --model-setting S1  --mode synthetic  --output synthetic_req_rate_a10g_S1_abort.jsonl

## a100-80g S4
# abort
# python launch_server.py  --device a100  --model-setting S4  --backend dm  --num-adapter 400  --num-token 65000 --enable-abort
python ../../../run_exp.py  --backend dm  --suite a100-80-S4-cv  --model-setting S4  --mode synthetic  --output synthetic_cv_a100-80_S4_abort.jsonl