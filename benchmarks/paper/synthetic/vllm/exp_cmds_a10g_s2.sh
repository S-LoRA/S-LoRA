## a100-80 S2

nvidia-cuda-mps-control -d

# vllm
python launch_server.py  --device a10g --model-setting S2  --backend vllm-packed  --num-adapter 1
python ../../../run_exp.py  --backend vllm-packed  --suite a10g-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a10g_S2_vllm.jsonl &&

