# python launch_server.py --dummy

# python run_exp.py --debug --suite increase
# 
# python run_exp.py --debug --suite on_off
# 
# python run_exp.py --debug --suite unbalance

# ---------------------------------

# python launch_server.py --dummy --fair 1 --fair 2

# python run_exp.py --debug --suite diff_slo

# ---------------------------------

# python launch_server.py --dummy --scheduler naive_fair

python run_exp.py --debug --suite increase --output all_results_increase_naive.jsonl

python run_exp.py --debug --suite on_off --output all_results_on_off_naive.jsonl

python run_exp.py --debug --suite unbalance --output all_results_unbalance_naive.jsonl

