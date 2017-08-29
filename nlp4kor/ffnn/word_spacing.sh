#!/usr/bin/env bash
PROGRAM="word_spacing.py"
PROJECT_DIR=${HOME}'/workspace/nlp4kor'

if [ "$#" -ne 3 ]; then
    echo "[Usage] $0 n_train ngram"
    echo "[Usage] $0 10000 6"
    echo "[Usage] $0 10000 4"
    exit
fi

n_train=$1
left_gram=$2
right_gram=$3

#git --work-tree=${PROJECT_DIR} --git-dir=${PROJECT_DIR}/.git pull

echo "pkill -f ${PROGRAM}"
pkill -f ${PROGRAM}

echo "rm -f logs/${PROGRAM}.$1.$2.$3.log"
rm -f logs/${PROGRAM}.$1.$2.$3.log
rm -f logs/${PROGRAM}.$1.$2.$3.error.log

#echo "rm -f logs/${PROGRAM}.*"
#rm -f logs/${PROGRAM}.*

echo "python3 ./${PROGRAM} $n_train $ngram >/dev/null 2>&1 &"
nohup python3 ./${PROGRAM} ${n_train} ${ngram} >/dev/null 2>&1 &
