#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
logfile=../../log/log.train_debug_${now}

srun --partition=MIA -w SH-IDC1-10-5-30-219 -n1 --job-name=train_B python -u train.py ../../configs/resnet18_crf_rs.json ../../model/crf_rs /2>&1 | tee ${logfile}
