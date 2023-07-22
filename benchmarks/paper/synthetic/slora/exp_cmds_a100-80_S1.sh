## a100-80 S1

# exp for the main table
# python launch_server.py  --device a100-80  --model-setting S1  --backend slora  --num-adapter 100  --num-token 125000
python ../../../run_exp.py  --backend slora  --suite a100-80-table  --model-setting S1  --mode synthetic  --output synthetic_num_adapters_a100-80_S1_slora_table.jsonl --append

