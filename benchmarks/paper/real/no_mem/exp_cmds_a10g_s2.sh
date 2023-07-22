## a10G S2
# slora-no-mem
# python launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 100  --num-token 11000 --no-mem
# python ../../../run_exp.py  --backend slora  --suite a10g-req-rate-real --model-setting S2  --mode real  --output real_req_rate_a10g_S2_no_mem.jsonl

# req rate = 2
# python launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 100  --num-token 10000 --no-mem
python ../../../run_exp.py  --backend slora  --suite a10g-req-rate-real --model-setting S2  --mode real  --output real_req_rate_a10g_S2_no_mem.jsonl --append
