## a10g S2
# slora
# python launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 14500
# python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_slora.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a10g-alpha  --model-setting S2  --mode synthetic  --output synthetic_alpha_a10g_S2_slora.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a10g-cv  --model-setting S2  --mode synthetic  --output synthetic_cv_a10g_S2_slora.jsonl
# python ../../../run_exp.py  --backend slora  --suite a10g-req-rate  --model-setting S2  --mode synthetic  --output synthetic_req_rate_a10g_S2_slora.jsonl

# ## a10g S1
# # slora
# python launch_server.py  --device a10g  --model-setting S1  --backend slora  --num-adapter 200  --num-token 14500
python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S1  --mode synthetic  --output synthetic_num_adapters_a10g_S1_slora.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a10g-alpha  --model-setting S1  --mode synthetic  --output synthetic_alpha_a10g_S1_slora.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a10g-cv  --model-setting S1  --mode synthetic  --output synthetic_cv_a10g_S1_slora.jsonl &&
python ../../../run_exp.py  --backend slora  --suite a10g-req-rate  --model-setting S1  --mode synthetic  --output synthetic_req_rate_a10g_S1_slora.jsonl

## a100-80g S4
# slora
# python launch_server.py  --device a100  --model-setting S4  --backend slora  --num-adapter 1000  --num-token 65000
# python ../../../run_exp.py  --backend slora  --suite a100-80-S4-num-adapter  --model-setting S4  --mode synthetic  --output synthetic_num_adapters_a100_80_S4_slora.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a100-alpha  --model-setting S4  --mode synthetic  --output synthetic_alpha_a100_80_S4_slora.jsonl &&
python ../../../run_exp.py  --backend slora  --suite a100-80-S4-cv  --model-setting S4  --mode synthetic  --output synthetic_cv_a100-80_S4_slora.jsonl
# python ../../../run_exp.py  --backend slora  --suite a100-80-S4-req-rate  --model-setting S4  --mode synthetic  --output synthetic_req_rate_a100_80_S4_slora.jsonl