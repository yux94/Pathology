#!/usr/bin/env sh
logfile=./log.FROC

srun -p MIA python -u Evaluation_FROC.py 2>&1 | tee ${logfile}
