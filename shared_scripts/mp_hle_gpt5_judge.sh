
export OPENAI_API_KEY=""

export BASE_URL="http://localhost:8000/v1/chat/completions"

output_dir=outputs/async-gpt-5-250807

#for output_dir in "outputs/async-gpt-5-250807-run1" "outputs/async-gpt-5-250807-run2"; do
output_dir=outputs/async-gpt-5-250807-run3
python main_judge_mp.py  \
  --dataset hle_math \
  --judge_method self \
  --baseline workflow_search --model gpt-5 --min_sample 0 --max_sample 199 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 64

for judge_method in "cot" "cot-sc" "debate" "reflexion"; do
  python main_judge_mp.py  \
  --dataset hle_math \
  --judge_method $judge_method \
  --baseline workflow_search --model gpt-5 --min_sample 0 --max_sample 199 --max_response_per_sample 9 \
  --save_dir $output_dir --num_workers 64
done
#done