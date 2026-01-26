export OPENAI_API_KEY=""
export BASE_PORT=8000

python async_main_question.py  \
  --dataset workflow_search/browsecomp-plus \
  --option plan \
  --meta_model gpt-5 \
  --node_model gpt-5 \
  --verifier_model gpt-4o_chatgpt --blocks COT COT_SC Reflexion LLM_debate \
  --use_oracle_verifier --defer_verifier --n_generation 5 --save_dir outputs/async-gpt-5-250807-run1 --max_workers 32 --max_tokens 131072


#python async_main_question.py  \
#  --dataset workflow_search/browsecomp-plus \
#  --option plan \
#  --meta_model gpt-5 \
#  --node_model gpt-5 \
#  --verifier_model gpt-4o_chatgpt --blocks COT COT_SC Reflexion LLM_debate \
#  --use_oracle_verifier --defer_verifier --n_generation 5 --save_dir outputs/async-gpt-5-250807-run2 --max_workers 32 --max_tokens 131072
