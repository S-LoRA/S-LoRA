## a10G S2
# bmm
# python launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 100  --num-token 6000 --bmm
python ../../../run_exp.py  --backend slora  --suite a10g-req-rate-real --model-setting S2  --mode real  --output real_req_rate_a10g_S2_bmm.jsonl

