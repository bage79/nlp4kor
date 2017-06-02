#!/usr/bin/env bash
PROGRAM="word_spacing.py"
PROJECT_DIR=${HOME}'/workspace/nlp4kor'

git --work-tree=${PROJECT_DIR} --git-dir=${PROJECT_DIR}/.git pull

echo "pkill -f ${PROGRAM}"
pkill -f ${PROGRAM}

echo "rm -f logs/${PROGRAM}.*"
rm -f logs/${PROGRAM}.*

echo "python3 ./${PROGRAM} $1 >/dev/null 2>&1 &"
nohup python3 ./${PROGRAM} $1 >/dev/null 2>&1 &
