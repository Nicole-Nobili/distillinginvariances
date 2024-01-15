# distillinginvariances

##Installation & running

clone the repo
```bash
   git clone https://github.com/Nicole-Nobili/distillinginvariances.git
```

install requirements
```bash
cd "path_to_repo"
pip install -r requirements.txt
```

NOTE: installing deepspeed on a Windows environment will be cumbersome. Without deepspeed, you will still be able to replicate all shift invariance results, apart from running calc_flops.py. 

To exactly replicate the shift invariance experiments: run the script experiments.py (without any arguments). This script will produce a solutions_processed.xlsx excel file with all metrics and sensitivities for each metric.