import argparse
import os
import time
from scienceworld import ScienceWorldEnv
import json
from eval_utils import findValidActionNew, load_variation
import tiktoken
from utils import *


import logging
from logging import INFO, WARN

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

# Call language model
def llm_gpt(prompt, stop=["\n"], model_name="gpt-40-1"):
    config = load_config('./baseline_config.ini')
    return get_gpt_response(
        config = config,
        gpt_prompt=prompt,
        model=model_name,
        temperature=0,
        max_token=50
    )
    # return get_azure_response(
    #     config = config,
    #     gpt_prompt=prompt,
    #     model=model_name,
    #     temperature=0,
    #     max_token=50
    # )

def get_file_name(args, task_num):
    if (len(args["output_path"]) > 0):
        args["output_path"] = args["output_path"] + "/"

        # Make path if it doesn't exist
        if (not os.path.exists(args['output_path'])):
            os.makedirs(args["output_path"])

    filenameOutPrefixSeed = args["output_path"] + "task" + str(task_num)

    return filenameOutPrefixSeed

# Example user input console, to play through a game.
def eval(args, task_num, logger):
    # Initialize environment
    env = ScienceWorldEnv("", args["jar_path"], envStepLimit=args["env_step_limit"])
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    env.load(taskName, 0, args['simplification_str'])
    variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num)

    # Load init prompt
    with open(args["prompt_file"], 'r') as f:
        d = json.load(f)
    
    # Load encoding tool to count token numbers
    encoding = tiktoken.encoding_for_model('gpt-4')

    scores = []

    for variation in variations:
        env.load(taskName, variation, args["simplification_str"], generateGoldPath=True)
        task_description = env.taskdescription()[18:]
        recent_actions = ["look around"]

        obs, info = env.reset()

        done = False
        score = 0.0
        last_score = 0.0
        step = 0

        max_steps = args["env_step_limit"] * 2

        init_prompt = 'Interact with a household to solve a task. Here is an example.\n' + d[str(task_num)]
        prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'

        max_len = 8192

        while not done:        
            while len(encoding.encode(init_prompt + prompt)) > max_len - 60:
                index1 = init_prompt.find('>')

                if index1 == -1:
                    index1_prompt = prompt.find('>')
                    index2_prompt = prompt.find('>', index1_prompt + 1)
                    prompt = prompt[:index1_prompt] + prompt[index2_prompt:]
                else:
                    index2 = init_prompt.find('>', index1 + 1)
                    if index2 == -1:
                        init_prompt = init_prompt[:index1]
                    else:
                        init_prompt = init_prompt[:index1] + init_prompt[index2:]

            logger.info("Prompt: " + init_prompt + prompt)
            action = llm_gpt(init_prompt + prompt, stop=['\n'], model_name=args["model_name"]).strip()

            if action.startswith('think:'):
                obs = 'OK.'
            else:
                action = findValidActionNew([action], env, info['look'], recent_actions, None, logger)
                obs, reward, done, info = env.step(action)

                score = info['score']

                if score < 0:
                    if args["no_stop"]:
                        done = True
                        score = last_score
                    else:
                        done = True
                        score = 0
                last_score = score
            
            obs = clean(obs)
            prompt += f' {action}\n{obs}\n>'
            recent_actions.append(action) 
            
            logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
            logger.info("Obs: " + obs)
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            if (step >= max_steps) or done:
                break

            logger.info("Recent Actions: " + str(recent_actions))

            if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                break

        env.storeRunHistory(variation, notes={'mode': "react_baseline", 'lm': None})
        env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"])

        scores.append(score)

        logger.info("Run completed...")
        logger.info("Scores: " + str(scores))
 
        time.sleep(2)

    env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"], forceSave=True)

    avg = sum(scores) / len(scores)
    logger.info("Average score: " + str(avg))

    with open(filenameOutPrefixSeed + "-score.txt", "a") as f:
        f.write("\n" + "Task name:" + taskName + " Scores: " + str(scores) + " Average score: " + str(avg) + " Args: " + str(args) + "\n")

    logger.info("Shutting down server...")
    # env.shutdown()

    logger.info("Completed.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar_path", type=str, default="") 
    parser.add_argument("--task_nums", default="0")  # use comma to split 
    parser.add_argument("--env_step_limit", type=int, default=100)
    parser.add_argument("--simplification_str", default="easy")
    parser.add_argument("--max_episode_per_file", type=int, default=9999)
    parser.add_argument("--set", default="test_mini")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--no_stop", action="store_true", default=False)
    parser.add_argument("--prompt_file", default="./prompts/react_prompts/prompt.jsonl")
    parser.add_argument("--model_name", default="gpt-4")

    args = parser.parse_args()
    params = vars(args)
    return params

def init_logger(args, task_num, log_level=INFO):
    filenameOutPrefixSeed = get_file_name(args, task_num)
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_dir = args["output_path"]
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime('%Y-%m-%d_%H-%M', time.localtime(now / 1000))
        filename = f"{filenameOutPrefixSeed}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger

def main():
    args = parse_args()
    print(args) 

    task_nums = args["task_nums"].split(",")
    for task_num in task_nums:
        logger = init_logger(args, task_num)
        logger.info(args)
        eval(args, int(task_num), logger)
        
if __name__ == "__main__":
    main()
