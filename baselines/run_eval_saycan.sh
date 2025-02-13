for task in 27
do
    python eval_agent_saycan.py \
        --task_nums $task \
        --set test_mini \
        --no_stop \
        --env_step_limit 100 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt_no_think.jsonl \
        --output_path saycan_logs/gpt-4-turbo-1 \
        --model_name gpt-40-1
done