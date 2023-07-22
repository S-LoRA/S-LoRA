## a100-80 S3

nvidia-cuda-mps-control -d

# vllm

# python launch_server.py  --device a100-80  --model-setting S3  --backend vllm-packed  --num-adapter 3 --vllm-mem-ratio 0.92
python ../../../run_exp.py  --backend vllm-packed  --suite a100-80-S3-num-adapter-vllm  --model-setting S3  --mode synthetic  --output synthetic_num_adapters_a100-80_S3_vllm.jsonl --append

