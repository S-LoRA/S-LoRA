# python launch_server.py

python run_exp.py --suite overload

python run_exp.py --suite proportional

python run_exp.py --suite increase

python run_exp.py --suite on_off_less

python run_exp.py --suite on_off_overload

python run_exp.py --suite poisson_on_off_overload

python run_exp.py --suite poisson_short_long

python run_exp.py --suite poisson_short_long_2

# ---------------------------------

# python launch_server.py --dummy --fair 1 --fair 2

# python run_exp.py --debug --suite diff_slo

# ---------------------------------

# python launch_server.py --dummy --scheduler naive_fair

# python run_exp.py --debug --suite increase --output all_results_increase_naive.jsonl
# 
# python run_exp.py --debug --suite on_off --output all_results_on_off_naive.jsonl
# 
# python run_exp.py --debug --suite unbalance --output all_results_unbalance_naive.jsonl

