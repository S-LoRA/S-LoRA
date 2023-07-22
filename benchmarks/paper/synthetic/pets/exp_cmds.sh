## a10g S2
# pets
# python launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 14500
python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_pets.jsonl &&
python ../../../run_exp.py  --backend slora  --suite a10g-alpha  --model-setting S2  --mode synthetic  --output synthetic_alpha_a10g_S2_pets.jsonl &&
python ../../../run_exp.py  --backend slora  --suite a10g-cv  --model-setting S2  --mode synthetic  --output synthetic_cv_a10g_S2_pets.jsonl &&
python ../../../run_exp.py  --backend slora  --suite a10g-req-rate  --model-setting S2  --mode synthetic  --output synthetic_req_rate_a10g_S2_pets.jsonl

# python launch_server.py  --device a10g  --model-setting S1  --backend slora  --num-adapter 200  --num-token 14500
# python ../../../run_exp.py  --backend slora  --suite a10g-num-adapter  --model-setting S1  --mode synthetic  --output synthetic_num_adapters_a10g_S1_pets.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a10g-alpha  --model-setting S1  --mode synthetic  --output synthetic_alpha_a10g_S1_pets.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a10g-cv  --model-setting S1  --mode synthetic  --output synthetic_cv_a10g_S1_pets.jsonl &&
# python ../../../run_exp.py  --backend slora  --suite a10g-req-rate  --model-setting S1  --mode synthetic  --output synthetic_req_rate_a10g_S1_pets.jsonl