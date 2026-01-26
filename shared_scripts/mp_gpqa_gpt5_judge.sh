
export OPENAI_API_KEY=""

export BASE_URL="http://localhost:8000/v1/chat/completions"


for output_dir in "outputs/async-gpt-5-250807-run1" "outputs/async-gpt-5-250807-run2"; do
python main_judge_mp.py  \
  --dataset gpqa_diamond \
  --judge_method self \
  --baseline workflow_search --model gpt-5 --min_sample 32 --max_sample 197 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 8

for judge_method in "cot" "cot-sc" "debate" "reflexion"; do
  python main_judge_mp.py  \
  --dataset gpqa_diamond \
  --judge_method $judge_method \
  --baseline workflow_search --model gpt-5 --min_sample 32 --max_sample 197 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 8
done
done