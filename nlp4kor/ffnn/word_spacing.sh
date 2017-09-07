#!/usr/bin/env bash
PROGRAM="word_spacing.py"
PROJECT_DIR=${HOME}'/workspace/nlp4kor'

if [ "$#" -ne 3 ]; then
    echo "[Usage] $0 n_train left_gram right_gram"
    echo "[Usage] $0 100 2 2"
    exit
fi

n_train=$1
left_gram=$2
right_gram=$3

#git --work-tree=${PROJECT_DIR} --git-dir=${PROJECT_DIR}/.git pull

echo "pkill -f ${PROGRAM}"
pkill -f ${PROGRAM}

echo "rm -f logs/${PROGRAM}.$1.$2.$3.*"
rm -f logs/${PROGRAM}.$1.$2.$3.*

echo "python3 ./${PROGRAM} $n_train $left_gram $right_gram >/dev/null 2>&1 &"
nohup python3 ./${PROGRAM} ${n_train} ${left_gram} ${right_gram} >/dev/null 2>&1 &
