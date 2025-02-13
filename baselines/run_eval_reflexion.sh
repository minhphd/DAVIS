for task in 18
do
    python eval_agent_reflexion.py \
        --task_nums $task \
        --set test_mini \
        --no_stop \
        --env_step_limit 80 \
        --simplification_str easy \
        --num_trials 4 \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path reflexion_logs/gpt-4-turbo-1 \
        --model_name gpt-40-1
done