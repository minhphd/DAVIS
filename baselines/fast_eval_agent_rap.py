import os
import sys
import yaml
import json
import numpy as np
import torch
import argparse
import logging
import tiktoken
import concurrent.futures
import pickle  # New: for saving memory in pickle format
from datetime import datetime
from eval_utils import findValidActionNew
from scienceworld import ScienceWorldEnv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from utils import *  # Assumes your helper functions (e.g. load_config, get_response) are here

# --------------------------
# Global Data & Helper Setup
# --------------------------

# Load the prompt examples for React (used by scienceworld_run_react)
with open('./prompts/rap_prompts/scienceworld_prompt.jsonl', 'r') as f:
    d = json.load(f)

# Dictionary mapping tasks to categories.
categories = {
    "boil": "State of Matter",
    "melt": "State of Matter",
    "freeze": "State of Matter",
    "change-the-state-of-matter-of": "State of Matter",
    "use-thermometer": "Measurement and Thermodynamics",
    "measure-melting-point-known-substance": "Measurement and Thermodynamics",
    "measure-melting-point-unknown-substance": "Measurement and Thermodynamics",
    "power-component": "Energy and Conductivity",
    "power-component-renewable-vs-nonrenewable-energy": "Energy and Conductivity",
    "test-conductivity": "Energy and Conductivity",
    "test-conductivity-of-unknown-substances": "Energy and Conductivity",
    "find-living-thing": "Living and Non-living Identification",
    "find-non-living-thing": "Living and Non-living Identification",
    "find-plant": "Living and Non-living Identification",
    "find-animal": "Living and Non-living Identification",
    "grow-plant": "Plant Growth",
    "grow-fruit": "Plant Growth",
    "chemistry-mix": "Chemistry and Mixing",
    "chemistry-mix-paint-secondary-color": "Chemistry and Mixing",
    "chemistry-mix-paint-tertiary-color": "Chemistry and Mixing",
    "lifespan-longest-lived": "Lifespan and Life Stages",
    "lifespan-shortest-lived": "Lifespan and Life Stages",
    "lifespan-longest-lived-then-shortest-lived": "Lifespan and Life Stages",
    "identify-life-stages-1": "Lifespan and Life Stages",
    "identify-life-stages-2": "Lifespan and Life Stages",
    "inclined-plane-determine-angle": "Inclined Planes and Friction",
    "inclined-plane-friction-named-surfaces": "Inclined Planes and Friction",
    "inclined-plane-friction-unnamed-surfaces": "Inclined Planes and Friction",
    "mendelian-genetics-known-plant": "Genetics",
    "mendelian-genetics-unknown-plant": "Genetics",
}

def clean(s):
    for tok in ['\n', '\t']:
        s = s.replace(tok, ' ')
    return s

def setup_logger(output_dir):
    log_filename = os.path.join(output_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)
    # Clear any existing handlers.
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def llm(prompt, stop=["\n"]):
    # Call the language model. (Ensure your get_response function is thread-safe.)
    text, _ = get_response(args.model, prompt, config=config)
    if stop:
        text = text.split('\n')[0]
    if text.startswith('>'):
        text = text[1:]
    if text.endswith('.'):
        text = text[:-1]
    return text.strip()

def process_ob(ob, env):
    if ob.startswith('You move to') or ob.startswith('You teleport to'):
        ob = env.look()
    return ob

def generate_embeddings(memory):
    logger = logging.getLogger("generate_embeddings")
    logger.info("Generating embeddings for %d memory items", len(memory))
    embeddings = {}
    for key in ['Task', 'Category', 'Plan', 'Actions']:
        if key == 'Actions':
            retrieve_info = [m[key].copy() for m in memory]
            for i in range(len(retrieve_info)):
                for j in range(len(retrieve_info[i])):
                    retrieve_info[i][j] = retrieve_info[i][j].strip()
            embeddings[key] = [model_embedding.encode(r) for r in retrieve_info]
        else:
            retrieve_info = [m[key] for m in memory]
            embeddings[key] = model_embedding.encode(retrieve_info)
    return embeddings

def generate_examples(query, memory, embeddings, k=3, for_plan=False, act_len=0, mode='act', key=''):
    if not memory:
        return []
    cos_scores_sum = []
    for key_tmp in ['Task', 'Category', 'Plan']:
        if query[key_tmp] == '':
            continue
        with torch.no_grad():
            query_embeddings = model_embedding.encode([query[key_tmp]])
        cos_scores = cos_sim(query_embeddings, embeddings[key_tmp])[0]
        cos_scores_sum.append(cos_scores.tolist())
    cos_scores_sum = torch.sum(torch.tensor(cos_scores_sum), 0)
    if for_plan:
        k = min(k, cos_scores_sum.shape[0])
        _, hits = torch.topk(cos_scores_sum, k=k)
        ret_examples = [
            'Your task is to: ' + memory[h]['Task'] + '\n> ' + memory[h]['Plan'] + '\n'
            for h in hits
        ]
        return ret_examples
    ret_scores = []
    ret_index = []
    with torch.no_grad():
        query_embeddings = model_embedding.encode([key])
    for emb in embeddings['Actions']:
        if key == '':
            ret_scores.append(0)
            ret_index.append(0)
            continue
        elif mode == 'act':
            log_embeddings = emb[::2]
        elif mode == 'obs':
            log_embeddings = emb[1::2]
        cos_scores = cos_sim(query_embeddings, log_embeddings).numpy()
        ret_scores.append(np.max(cos_scores))
        ret_index.append(np.argmax(cos_scores) * 2)
    ret_scores = torch.FloatTensor(ret_scores)
    k = min(k, (ret_scores + cos_scores_sum).shape[0])
    _, hits = torch.topk(ret_scores + cos_scores_sum, k=k)
    ret_examples = []
    for h in hits:
        part = (max(0, ret_index[h] - act_len), min(len(memory[h]['Actions']), ret_index[h] + act_len))
        ret_examples.append(
            'Task: ' + memory[h]['Task'] + '\nPlan: ' + memory[h]['Plan'] + '\n' +
            '\n'.join(memory[h]['Actions'][part[0]:part[1]]) + '\n'
        )
    return ret_examples

# --------------------------
# Experiment Functions
# --------------------------
def scienceworld_run_golden(category, env, logger, task_num, to_print=False):
    """
    Runs a React-based approach with the ScienceWorld environment.
    """
    logger.info("Starting scienceworld_run_react with category: %s", category)
    
    task_description = env.taskdescription()[18:]
    obs, info = env.reset()
    logger.debug("Environment reset: obs='%s', info=%s", obs, info)
    
    init_prompt = 'Interact with a household to solve a task. Here is an example.\n' + d[str(task_num)]
    prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'
    recent_actions = ["look around"]
    actions = []
    done = False
    step = 0
    score = 0.0
    last_score = 0.0
    
    for action in env.getGoldActionSequence():    
        # action = findValidActionNew([action], env, info['look'], recent_actions, None, logger)
        obs, reward, done, info = env.step(action)
        score = info.get('score', score)
        logger.debug("Executed action: %s | Obs: %s, Reward: %s, Done: %s, Info: %s",
                        action, obs, reward, done, info)
        if score < 0:
            logger.warning("Received negative score (%s). Terminating with rollback.", score)
            done = True
            score = last_score
        last_score = score

        obs = clean(obs)
        actions.append('> ' + action)
        actions.append(obs)
        prompt += f' {action}\n{obs}\n>'
        
        if to_print:
            print(f'Act {step + 1}: {action}\nObs: {obs}')
        
        recent_actions.append(action)

        step += 1
        
        if done:
            logger.info("Terminating loop as done flag is True.")
            break

    inv_act_idx = np.where(np.array(actions) == 'Nothing happens.')[0]
    inv_act_idx = np.append(inv_act_idx, inv_act_idx - 1)
    actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]

    data = {
        'Task': task_description,
        'Category': category,
        'Plan': env.getTaskDescription(),
        'Actions': actions,
    }

    logger.info("Run completed. Final score: %s", score)
    logger.info("Actions taken: %s", actions)
    return (score, data) if done else (0, '')


def scienceworld_run_react(category, env, logger, task_num, to_print=False):
    """
    Runs a React-based approach with the ScienceWorld environment.
    """
    logger.info("Starting scienceworld_run_react with category: %s", category)
    
    task_description = env.taskdescription()[18:]
    obs, info = env.reset()
    logger.debug("Environment reset: obs='%s', info=%s", obs, info)
    
    init_prompt = 'Interact with a household to solve a task. Here is an example.\n' + d[str(task_num)]
    prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'
    recent_actions = ["look around"]
    actions = []
    done = False
    step = 0
    score = 0.0
    last_score = 0.0

    while not done:
        encoding = tiktoken.encoding_for_model('gpt-4')
        while len(encoding.encode(init_prompt + prompt)) > 8192 - 60:
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
        full_input = init_prompt + prompt
        action = llm(full_input, stop=['\n']).strip()
        logger.info("Step %d: Generated action: %s", step + 1, action)

        if action.startswith('think:'):
            obs = 'OK.'
        else:
            # action = findValidActionNew([action], env, info['look'], recent_actions, None, logger)
            obs, reward, done, info = env.step(action)
            score = info.get('score', score)
            logger.debug("Executed action: %s | Obs: %s, Reward: %s, Done: %s, Info: %s",
                         action, obs, reward, done, info)
            if score < 0:
                logger.warning("Received negative score (%s). Terminating with rollback.", score)
                done = True
                score = last_score
            last_score = score

        obs = clean(obs)
        actions.append('> ' + action)
        actions.append(obs)
        prompt += f' {action}\n{obs}\n>'
        
        if to_print:
            print(f'Act {step + 1}: {action}\nObs: {obs}')
        
        recent_actions.append(action)
        if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
            logger.warning("Detected action loop after %d steps, stopping early.", step + 1)
            break

        step += 1
        if step > args.num_steps:
            logger.info("Maximum steps exceeded")
            break
        
        if done:
            logger.info("Terminating loop as done flag is True.")
            break

    if 'think:' in actions[0]:
        plan = actions[0].split('think: ')[1].strip()
    else:
        plan = actions[0]

    inv_act_idx = np.where(np.array(actions) == 'Nothing happens.')[0]
    inv_act_idx = np.append(inv_act_idx, inv_act_idx - 1)
    actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]

    data = {
        'Task': task_description,
        'Category': category,
        'Plan': plan,
        'Actions': actions,
    }

    logger.info("Run completed. Final score: %s", score)
    logger.info("Actions taken: %s", actions)
    return (score, data) if done else (0, '')

def scienceworld_run_rap(ob, category, memory, embeddings, env, logger, to_print=True):
    """
    Runs the RAP-based approach with retrieval and planning for the ScienceWorld environment.
    """
    logger.info("Starting scienceworld_run_rap with task: %s", ob.split('\n')[0])
    
    if to_print:
        print(ob)
    ob_prompt = 'Here is the task information.\n' + ob.split('\n')[0] + '\n'
    target_task = ob.split('\n')[0]

    examples = generate_examples({'Task': target_task, 'Category': category, 'Plan': ''},
                                 memory, embeddings, k=3, for_plan=True)
    examples_text = 'Here are examples.\n' + ''.join(examples)
    target_prompt = ('\nHere is the task. Please make a plan from the examples.\n'
                     'Your task is to: ' + target_task + '\n> think: To solve the task,')
    full_prompt = examples_text + target_prompt
    plan = llm(full_prompt)
    plan = 'To solve the task, ' + plan.split('.')[0] + '.'
    
    target_prompt = 'Here is the task. Please make an action from the examples.\nTask : ' + target_task + '\nPlan : ' + plan + '\n'
    
    data = {
        'Task': target_task,
        'Category': category,
        'Plan': plan,
        'Actions': '',
    }
    actions = []
    search_object = ''
    reasoning = ''
    last_score = 0
    ret_examples = generate_examples(data, memory, embeddings, k=4, act_len=20)
    
    for i in range(1, args.num_steps):
        examples = 'Here are examples.\n' + ''.join(ret_examples)
        full_prompt = ob_prompt + examples + target_prompt + '\n'.join(data['Actions']) + '\n>'
        action = llm(full_prompt)
        logger.info("Step %d: Generated action: %s", i, action)
            
        observation, _, done, info = env.step(action)
        reward = info.get('reward', 0)
        fail = (info.get('score', 0) == -100)
        logger.debug("Step %d: Executed action: %s | Obs: %s, Reward: %s, Done: %s, Info: %s",
                     i, action, observation, reward, done, info)
        
        observation = process_ob(observation, env)
        if action.startswith('think:'):
            observation = 'OK.'
        
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            
        if 'think:' in action:
            full_prompt_key = ('Here are examples.\n' + ret_key_examples_str +
                               '\nHere is the task. Please make a plan from the examples.\n' + action + '\n>')
            retrieve_key = llm(full_prompt_key)
            logger.info("Step %d: Retrieved key: %s", i, retrieve_key)
            if 'search:' in retrieve_key:
                search_object = retrieve_key.split('search:')[1].strip()
                ret_examples = generate_examples(data, memory, embeddings, k=8, act_len=10, mode='obs', key=search_object)
            elif 'action:' in retrieve_key:
                reasoning = retrieve_key.split('action:')[1].strip()
                ret_examples = generate_examples(data, memory, embeddings, k=4, act_len=20, mode='act', key=reasoning)

        actions.append('> ' + action)
        actions.append(observation)
        data['Actions'] = actions[-10:]
        
        if fail:
            logger.error("Step %d: Action resulted in failure. Exiting with last score: %s", i, last_score)
            return last_score, ''
        
        if done:
            inv_act_idx = np.where(np.array(actions) == 'Nothing happens.')[0]
            inv_act_idx = np.append(inv_act_idx, inv_act_idx - 1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            data['Actions'] = actions
            logger.info("Task completed successfully with final reward: %s", reward)
            return reward, data
        
        last_score = info.get('score', last_score)
    
    logger.info("Reached maximum steps with final score: %s", info.get('score', last_score))
    return info.get('score', last_score), ''

# --------------------------
# Agent Runner: Process a Subset of Tasks with One Environment
# --------------------------

def run_agent_experiments(trial, task_indices, mode, memory, embeddings, logger, simplification):
    """
    Creates one ScienceWorldEnv and processes a subset of tasks sequentially.
    Returns a list of results for the assigned task indices.
    """
    env = ScienceWorldEnv()
    results = []
    task_names = env.getTaskNames()
    for task_index in task_indices:
        task = task_names[task_index]
        category = categories[task]
        env.load(task, variationIdx=trial, simplificationStr=simplification, generateGoldPath=True)
        env.reset()
        if mode == "trainning":
            reward, mem_data = scienceworld_run_golden(category, env, logger, task_index)
        else:
            reward, mem_data = scienceworld_run_rap(env.getTaskDescription(), category, memory, embeddings, env, logger)
        results.append((trial, task, mode, reward, mem_data))
    del env
    return results

# --------------------------
# Main: Use ThreadPoolExecutor with Fixed Number of Agent Environments,
# Dump Results Immediately, and Save Memory as Pickle for Training Tasks
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=8, help="The number of trials")
    parser.add_argument("--num_steps", type=int, default=50, help="The number of steps")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The model name")
    parser.add_argument("--output", type=str, default="./rap_logs", help="The output folder")
    parser.add_argument("--emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        choices=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"],
                        help="The model name")
    parser.add_argument("--num_agents", type=int, default=5,
                        help="Number of agent environments to spawn concurrently")
    # New arguments for checkpointing/resume:
    parser.add_argument("--resume", action="store_true", help="Resume from a checkpoint if available")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint.pkl",
                        help="Filename to store checkpoint data")
    args_parsed = parser.parse_args()

    global args, config  # Make these available for llm() and other functions.
    args = args_parsed
    config = load_config('./baseline_config.ini')
    
    logger = setup_logger(args.output)
    logger.info("Successfully set up logger and loaded config.")
    
    global model_embedding
    model_embedding = SentenceTransformer(args.emb_model)
    
    tasks_list = list(categories.keys())
    simplification = 'easy'
    
    os.makedirs(args.output, exist_ok=True)
    
    # Read retrieval key examples for RAP (if needed)
    ret_key_examples = open('./prompts/rap_prompts/retrieval_prompt.txt').readlines()
    global ret_key_examples_str
    ret_key_examples_str = ''.join(ret_key_examples)
    
    # Define the checkpoint file path.
    checkpoint_file = os.path.join(args.output, args.checkpoint_file)
    
    # If resuming, load checkpoint data.
    if args.resume and os.path.exists(checkpoint_file):
        print('loading')
        with open(checkpoint_file, "rb") as cp_f:
            checkpoint_data = pickle.load(cp_f)
        start_trial = checkpoint_data.get("last_trial", -1) + 1
        all_results = checkpoint_data.get("all_results", [])
        current_memory = checkpoint_data.get("current_memory", [])
        logger.info("Resuming from trial %d", start_trial)
    else:
        start_trial = 0
        all_results = []
        current_memory = []
    
    memory = []  # For evaluation mode.
    
    results_file = os.path.join(args.output, "results.json")
    
    # Loop over trials (starting at start_trial if resuming)
    for trial in range(start_trial, args.num_trials):
        mode = "trainning" if trial < 5 else "evaluating"
        logger.info("### Trial %d Mode: %s ###", trial+1, mode)
        
        if mode == "evaluating":
            memory = current_memory[:]  # Use training memory.
            embeddings = generate_embeddings(memory)
        else:
            embeddings = None  # Not used in training mode.
        
        # Partition the tasks among agents.
        tasks_to_run = list(range(len(tasks_list)))[:10]
        # tasks_to_run = [0]
        num_agents = args.num_agents
        partitions = [tasks_to_run[i::num_agents] for i in range(num_agents)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = []
            for partition in partitions:
                futures.append(executor.submit(run_agent_experiments, trial, partition, mode, memory, embeddings, logger, simplification))
            for future in concurrent.futures.as_completed(futures):
                agent_results = future.result()
                for result in agent_results:
                    trial_val, task, mode_val, reward, mem_data = result
                    result_entry = {
                        "trial": trial_val,
                        "task": task,
                        "mode": mode_val,
                        "reward": reward
                    }
                    all_results.append(result_entry)
                    
                    # Dump current results to JSON after each task completion.
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2)
                    
                    # For training tasks, if memory data is returned, update current_memory
                    # and save it as a pickle file for easier loading later.
                    if mode_val == "trainning" and mem_data != '':
                        current_memory.append(mem_data)
                        # mem_pickle_file = os.path.join(args.output, f"memory_trial_{trial}_task_{task}.pkl")
                        # with open(mem_pickle_file, "wb") as pf:
                        #     pickle.dump(current_memory, pf)
                        # logger.info("Saved memory for task '%s' trial %d to pickle file: %s", task, trial_val, mem_pickle_file)
                    
                    if mode_val == "evaluating":
                        logger.info("Task '%s' variation %d evaluated with reward: %s", task, trial_val, reward)
                    else:
                        logger.info("Task '%s' variation %d training completed with reward: %s", task, trial_val, reward)
        
        # Save a checkpoint after each trial.
        checkpoint_data = {
            "last_trial": trial,
            "all_results": all_results,
            "current_memory": current_memory
        }
        with open(checkpoint_file, "wb") as cp_f:
            pickle.dump(checkpoint_data, cp_f)
        logger.info("Checkpoint saved after trial %d", trial)
    
    logger.info("All trials completed.")

if __name__ == "__main__":
    main()
