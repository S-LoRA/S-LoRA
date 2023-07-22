# python ../../run_exp_peft.py --model-setting S1 --suite a10g --breakdown --output num_adapter_a10g_S1_peft.jsonl --append
python ../../run_exp_peft.py --model-setting S2 --suite a10g --breakdown --output num_adapter_a10g_S2_peft.jsonl

# against cv
python ../../run_exp_peft.py --model-setting S1 --suite diff_cv --breakdown --output num_cv_a10g_S1_peft.jsonl 
python ../../run_exp_peft.py --model-setting S2 --suite diff_cv --breakdown --output num_cv_a10g_S2_peft.jsonl 

# against alpha
python ../../run_exp_peft.py --model-setting S1 --suite diff_alpha --breakdown --output num_alpha_a10g_S1_peft.jsonl 
python ../../run_exp_peft.py --model-setting S2 --suite diff_alpha --breakdown --output num_alpha_a10g_S2_peft.jsonl 

# against alpha
python ../../run_exp_peft.py --model-setting S1 --suite diff_req --breakdown --output num_req_a10g_S1_peft.jsonl 
python ../../run_exp_peft.py --model-setting S2 --suite diff_req --breakdown --output num_req_a10g_S2_peft.jsonl 

