# Requirements
* CUDA 11.8

# Install
```
conda create -n slora python=3.9
conda activate slora
pip install -r requirements.txt
pip install -e .
```
Check that we have triton==2.1.0

# Example Run
```
cd benchmarks
python launch_server.py --num-adapter 1 --device debug --backend dm
python run_exp.py --debug --backend dm --suite swap
```

# Test Correctness
```
cd test/test_e2e
python launch_server.py
python run_exp.py
```

# Plots
```
cd benchmarks/plot
python plot_main_synthetic.py
python plot_main_real.py
python plot_ablation_abort.py
python plot_cluster_ablation.py
```
