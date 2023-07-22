## a10g S1
# slora
python ../../launch_server.py  --device a10g  --model-setting S1  --backend slora  --num-adapter 200  --num-token 14500
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S1  --mode synthetic  --output ablation_mem_a10g_S1_dm.jsonl

# no-mem
python ../../launch_server.py  --device a10g  --model-setting S1  --backend slora  --num-adapter 200  --num-token 14500 --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S1  --mode synthetic  --output ablation_mem_a10g_S1_no_mem.jsonl

# no-mem (num_adapter = 50)
python ../../launch_server.py  --device a10g  --model-setting S1  --backend slora  --num-adapter 200  --num-token 13500 --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S1  --mode synthetic  --output ablation_mem_a10g_S1_no_mem.jsonl --append


## a10g S2
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 14500
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_mem_a10g_S2_dm.jsonl

# no-mem
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 14500 --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_mem_a10g_S2_no_mem.jsonl

# no-mem (num_adapter = 10)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 13500 --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_mem_a10g_S2_no_mem.jsonl --append

# no-mem (num_adapter = 25)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 10000 --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_mem_a10g_S2_no_mem.jsonl --append

# no-mem (num_adapter = 50)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 9500 --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_mem_a10g_S2_no_mem.jsonl --append


# no-mem (num_adapter = 200)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 9000 --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_mem_a10g_S2_no_mem.jsonl --append

# bmm + no-mem 
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 13000  --bmm  --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_bmm_a10g_S2_no_mem.jsonl

# bmm + no-mem (num_adapter = 10)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 9000  --bmm  --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_bmm_a10g_S2_no_mem.jsonl --append

# bmm + no-mem (num_adapter = 25)
python ../../launch_server.py  --device a10g  --model-setting S2  --backend slora  --num-adapter 200  --num-token 7000  --bmm  --no-mem-pool
python ../../run_exp.py  --backend slora  --suite ablation-no-mem  --model-setting S2  --mode synthetic  --output ablation_bmm_a10g_S2_no_mem.jsonl --append

## a10g S3


## a10g S4
