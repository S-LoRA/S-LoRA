## a100 S2
# dm (batch_num_adapters = 1)
python ../../launch_server.py --device a10g --model-setting S2  --backend dm  --num-adapter 32 --num-token 14500 --batch-num-adapters 1
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S2  --mode synthetic  --output ablation_cluster_size_1_a100_S2.jsonl

# dm (batch_num_adapters = 2)
python ../../launch_server.py --device a10g --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 2
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S2  --mode synthetic  --output ablation_cluster_size_2_a100_S2.jsonl

# dm (batch_num_adapters = 4)
python ../../launch_server.py  --device a10g --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 4
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S2  --mode synthetic  --output ablation_cluster_size_4_a100_S2.jsonl

# dm (batch_num_adapters = 8)
python ../../launch_server.py  --device a10g --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 8
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S2  --mode synthetic  --output ablation_cluster_size_8_a100_S2.jsonl

# dm (batch_num_adapters = 32)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 32
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S2  --mode synthetic  --output ablation_cluster_size_32_a100_S2.jsonl

## a100 S4
# dm (batch_num_adapters = 1)
python ../../launch_server.py --device a10g --model-setting S4  --backend dm  --num-adapter 32 --num-token 14500 --batch-num-adapters 1
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S4  --mode synthetic  --output ablation_cluster_size_1_a100_S4.jsonl

# dm (batch_num_adapters = 2)
python ../../launch_server.py --device a10g --model-setting S4  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 2
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S4  --mode synthetic  --output ablation_cluster_size_2_a100_S4.jsonl

# dm (batch_num_adapters = 4)
python ../../launch_server.py  --device a10g --model-setting S4  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 4
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S4  --mode synthetic  --output ablation_cluster_size_4_a100_S4.jsonl

# dm (batch_num_adapters = 8)
python ../../launch_server.py  --device a10g --model-setting S4  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 8
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S4  --mode synthetic  --output ablation_cluster_size_8_a100_S4.jsonl

# dm (batch_num_adapters = 32)
python ../../launch_server.py  --device a10g  --model-setting S4  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 32
python ../../run_exp.py  --backend dm  --suite ablation-cluster --model-setting S4  --mode synthetic  --output ablation_cluster_size_32_a100_S4.jsonl

## a100 S2-cv
python ../../launch_server.py --device a10g --model-setting S2  --backend dm  --num-adapter 32 --num-token 14500 --batch-num-adapters 1
python ../../run_exp.py  --backend dm  --suite ablation-cluster-cv --model-setting S2  --mode synthetic  --output ablation_cluster_cv_size_1_a100_S2.jsonl

# dm (batch_num_adapters = 2)
python ../../launch_server.py --device a10g --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 2
python ../../run_exp.py  --backend dm  --suite ablation-cluster-cv --model-setting S2  --mode synthetic  --output ablation_cluster_cv_size_2_a100_S2.jsonl

# dm (batch_num_adapters = 4)
python ../../launch_server.py  --device a10g --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 4
python ../../run_exp.py  --backend dm  --suite ablation-cluster-cv --model-setting S2  --mode synthetic  --output ablation_cluster_cv_size_4_a100_S2.jsonl

# dm (batch_num_adapters = 8)
python ../../launch_server.py  --device a10g --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 8
python ../../run_exp.py  --backend dm  --suite ablation-cluster-cv --model-setting S2  --mode synthetic  --output ablation_cluster_cv_size_8_a100_S2.jsonl

# dm (batch_num_adapters = 32)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend dm  --num-adapter 32  --num-token 14500 --batch-num-adapters 32
python ../../run_exp.py  --backend dm  --suite ablation-cluster-cv --model-setting S2  --mode synthetic  --output ablation_cluster_cv_size_32_a100_S2.jsonl