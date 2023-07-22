## a100-80 S2

nvidia-cuda-mps-control -d

# vllm
# python launch_server.py  --device a100-80  --model-setting S2  --backend vllm-packed  --num-adapter 1
# python ../../../run_exp.py  --backend vllm-packed  --suite a100-80-num-adapter-vllm  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_vllm.jsonl
# 
# python launch_server.py  --device a100-80  --model-setting S2  --backend vllm-packed  --num-adapter 2
# python ../../../run_exp.py  --backend vllm-packed  --suite a100-80-num-adapter-vllm  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_vllm.jsonl --append
# 
# python launch_server.py  --device a100-80  --model-setting S2  --backend vllm-packed  --num-adapter 4 --vllm-mem-ratio 0.92
# python ../../../run_exp.py  --backend vllm-packed  --suite a100-80-num-adapter-vllm  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_vllm.jsonl --append

# python launch_server.py  --device a100-80  --model-setting S2  --backend vllm-packed  --num-adapter 5 --vllm-mem-ratio 0.92
python ../../../run_exp.py  --backend vllm-packed  --suite a100-80-S2-num-adapter-vllm  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_vllm.jsonl --append

