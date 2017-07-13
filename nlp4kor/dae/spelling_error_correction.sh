#!/usr/bin/env bash
PROGRAM="spelling_error_correction.py"
PROJECT_DIR=${HOME}'/workspace/nlp4kor'

if [ "$#" -ne 3 ]; then
    echo "[Usage] $0 n_train window_size noise_rate"
    echo "[Usage] $0 1000000 6 0.1"
    exit
fi

n_train=$1
window_size=$2
noise_rate=$3

#git --work-tree=${PROJECT_DIR} --git-dir=${PROJECT_DIR}/.git pull

echo "pkill -f ${PROGRAM}"
pkill -f ${PROGRAM}

echo "rm -f logs/${PROGRAM}.$1.$2.$3.log"
rm -f logs/${PROGRAM}.$1.$2.$3.log
rm -f logs/${PROGRAM}.$1.$2.$3.error.log

#echo "rm -f logs/${PROGRAM}.*"
#rm -f logs/${PROGRAM}.*

echo "python3 ./${PROGRAM} $n_train $window_size $noise_rate >/dev/null 2>&1 &"
nohup python3 ./${PROGRAM} $n_train $window_size $noise_rate >/dev/null 2>&1 &
