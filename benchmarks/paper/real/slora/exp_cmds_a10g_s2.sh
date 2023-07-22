## a10G S2
# slora
# python launch_server.py  --device a10g  --model-setting S2  --backend dm  --num-adapter 100  --num-token 14500
python ../../../run_exp.py  --backend dm  --suite a10g-req-rate-real --model-setting S2  --mode real  --output real_req_rate_a10g_S2_slora.jsonl --append
