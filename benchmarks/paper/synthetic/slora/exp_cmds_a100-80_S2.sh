## a100-80 S2
# slora (exp for the main plot)
# python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 500  --num-token 125000
# python ../../../run_exp.py  --backend slora  --suite a100-80-S2-num-adapter  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_slora.jsonl &&

# exp for the main table
# python launch_server.py  --device a100-80  --model-setting S2  --backend slora  --num-adapter 100 --num-token 125000
python ../../../run_exp.py  --backend slora  --suite a100-80-table  --model-setting S2  --mode synthetic  --output synthetic_num_adapters_a100-80_S2_slora_table.jsonl --append
