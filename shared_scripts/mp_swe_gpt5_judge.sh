#!/usr/bin/env bash
set -euo pipefail   # 关键：出错就退出；用到未定义变量也报错


export OPENAI_API_KEY=""

export BASE_URL="http://localhost:8000/v1/chat/completions"

#python main_judge_mp.py  \
#    --dataset swe \
#    --judge_method reflexion \
#    --baseline workflow_search --model gpt-5 --min_sample 0 --max_sample 267 --max_response_per_sample 9 \
#    --save_dir outputs/async-gpt-5-250807 --num_workers 64


for output_dir in "outputs/async-gpt-5-250807" "outputs/async-gpt-5-250807-run1" "outputs/async-gpt-5-250807-run2"; do
#output_dir="outputs/async-gpt-5-250807"
  python main_judge_mp.py  \
    --dataset swe \
    --judge_method self \
    --baseline workflow_search --model gpt-5 --min_sample 0 --max_sample 267 --max_response_per_sample 9 \
    --save_dir $output_dir --num_workers 16

  for judge_method in "cot" "cot-sc" "debate" "reflexion"; do
#    judge_method="reflexion"
    python main_judge_mp.py  \
    --dataset swe \
    --judge_method $judge_method \
    --baseline workflow_search --model gpt-5 --min_sample 0 --max_sample 267 --max_response_per_sample 9 \
    --save_dir $output_dir  --num_workers 16
  done
done