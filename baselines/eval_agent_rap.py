import os,sys
import yaml
import json
import numpy as np
from eval_utils import findValidActionNew
import tiktoken
from scienceworld import ScienceWorldEnv
import torch
import argparse
from utils import *
from datetime import datetime

# text embedding model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
 
def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s
 
# Function to set up the logger
def setup_logger(output_dir):
    log_filename = os.path.join(output_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplication
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
    text, _ = get_response(args.model, prompt, config=config)
    if stop:
        text = text.split('\n')[0]
    if len(text) > 0 and text[0]=='>':
        text = text[1:]
    if len(text) > 0 and text[-1]=='.':
        text = text[:-1]
    return text.strip()

def process_ob(ob, env):
    if ob.startswith('You move to') or ob.startswith('You teleport to'):
        ob = env.look()
    return ob

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


def generate_embeddings(memory):
    print('num_retrieval',len(memory))
    embeddings = {}
    for key in ['Task', 'Category', 'Plan', 'Actions']:
        if key=='Actions':
            retrieve_info = [m[key].copy() for m in memory]
            for i in range(len(retrieve_info)):
                for j in range(len(retrieve_info[i])):
                    retrieve_info[i][j] = retrieve_info[i][j].strip()
            embeddings[key] = [model_embedding.encode(r) for r in retrieve_info]
            continue
        retrieve_info = [m[key] for m in memory]
        # extract embeddings
        embeddings[key] = model_embedding.encode(retrieve_info)
    return embeddings

def generate_examples(query, memory, embeddings, k=3, for_plan=False, act_len=0, mode='act', key=''):
    # similarity on task, category, and plan
    if not memory:
        return []
    
    cos_scores_sum = []
    for key_tmp in ['Task', 'Category', 'Plan']:
        if query[key_tmp]=='': continue
        with torch.no_grad():
            query_embeddings = model_embedding.encode([query[key_tmp]])
        cos_scores = cos_sim(query_embeddings, embeddings[key_tmp])[0]
        cos_scores_sum.append(cos_scores.tolist())
    cos_scores_sum = torch.sum(torch.tensor(cos_scores_sum), 0)
    # retrieve examples for overall plan
    if for_plan:
        _, hits = torch.topk(cos_scores_sum, k=k)
        # print(memory[h])
        # raise
        ret_examples = [ 'Your task is to: ' + memory[h]['Task'] + '\n> ' + memory[h]['Plan'] + '\n' for h in hits]
        return ret_examples
    # similarity on action or observation
    ret_scores=[]
    ret_index=[]
    with torch.no_grad():
        query_embeddings = model_embedding.encode([key])
    for emb in embeddings['Actions']:
        if key=='':
            ret_scores.append(0)
            ret_index.append(0)
            continue
        elif mode=='act':
            # pick up action embeddings
            log_embeddings = emb[::2]
        elif mode=='obs':
            # pick up observation embeddings
            log_embeddings = emb[1::2]
        cos_scores = cos_sim(query_embeddings, log_embeddings).numpy()
        ret_scores.append(np.max(cos_scores))
        ret_index.append(np.argmax(cos_scores)*2)
    ret_scores = torch.FloatTensor(ret_scores)
    # retrieve examples for action or action plan
    _, hits = torch.topk(ret_scores+cos_scores_sum, k=k)
    ret_examples = []
    for h in hits:
        part = (max(0,ret_index[h]-act_len),min(len(memory[h]['Actions']),ret_index[h]+act_len))
        ret_examples.append('Task: ' + memory[h]['Task'] + '\nPlan: ' + memory[h]['Plan'] + '\n' + '\n'.join(memory[h]['Actions'][part[0]:part[1]]) + '\n')
    return ret_examples

def scienceworld_run_react(category, env, logger, to_print=False):
    """
    Runs a React-based approach with the ScienceWorld environment.

    Args:
        category (str): The category of the task.
        env (ScienceWorldEnv): The ScienceWorld environment instance.
        logger (Logger): Logger instance to log actions and observations.
        to_print (bool): Whether to print outputs or not.

    Returns:
        tuple: The final score and a dictionary containing the task information.
    """
    logger.info("Starting scienceworld_run_react with category: %s", category)
    
    # Initialize task variables
    task_description = env.taskdescription()[18:]  # Get task description
    obs, info = env.reset()
    logger.debug("Environment reset: obs='%s', info=%s", obs, info)
    
    # Note: The following init_prompt uses undefined variables (e.g. d and task_num).
    # Make sure these variables are defined in your context.
    init_prompt = 'Interact with a household to solve a task. Here is an example.\n' + d[str(task_num)]
    prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'
    recent_actions = ["look around"]
    actions = []
    done = False
    step = 0
    score = 0.0
    last_score = 0.0

    while not done:
        # Ensure the combined prompt stays under the token limit for the LLM
        encoding = tiktoken.encoding_for_model('gpt-4')
        while len(encoding.encode(init_prompt + prompt)) > 8192 - 60:
            index1 = init_prompt.find('>')
            if index1 == -1:
                index1_prompt = prompt.find('>')
                index2_prompt = prompt.find('>', index1_prompt + 1)
                prompt = prompt[:index1_prompt] + prompt[index2_prompt:]
                # logger.debug("Trimmed prompt to reduce token count.")
            else:
                index2 = init_prompt.find('>', index1 + 1)
                if index2 == -1:
                    init_prompt = init_prompt[:index1]
                else:
                    init_prompt = init_prompt[:index1] + init_prompt[index2:]
                # logger.debug("Trimmed init_prompt to reduce token count.")

        # Generate an action using the LLM
        full_input = init_prompt + prompt
        action = llm(full_input, stop=['\n']).strip()
        logger.info("Step %d: Generated action: %s", step + 1, action)

        # Handle reasoning-only actions
        if action.startswith('think:'):
            obs = 'OK.'
        else:
            # Execute action in the environment
            obs, reward, done, info = env.step(action)
            score = info.get('score', score)
            logger.debug("Executed action: %s | Received obs: %s, reward: %s, done: %s, info: %s",
                         action, obs, reward, done, info)
            # Handle invalid or negative scores
            if score < 0:
                logger.warning("Received negative score (%s). Terminating with rollback.", score)
                done = True
                score = last_score  # Rollback to the last valid score
            last_score = score

        # Clean and log observation
        obs = clean(obs)
        actions.append('> ' + action)
        actions.append(obs)

        # Update the game prompt with the new action and observation
        prompt += f' {action}\n{obs}\n>'
        
        if to_print:
            print(f'Act {step + 1}: {action}\nObs: {obs}')
        
        # Check for loops or excessive steps
        recent_actions.append(action)
        if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
            logger.warning("Detected action loop after %d steps, stopping early.", step + 1)
            break

        step += 1
        if done:
            logger.info("Terminating loop as done flag is True.")
            break

    # Extract the plan from the first action if it is a reasoning action
    if 'think:' in actions[0]:
        plan = actions[0].split('think: ')[1].strip()
    else:
        plan = actions

    # Remove invalid actions or observations (e.g., 'Nothing happens.')
    import numpy as np
    inv_act_idx = np.where(np.array(actions) == 'Nothing happens.')[0]
    inv_act_idx = np.append(inv_act_idx, inv_act_idx - 1)
    actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]

    # Prepare task data
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

    Args:
        ob (str): The initial observation or task description.
        category (str): The category of the task.
        memory: Memory storage for examples.
        embeddings: Embeddings to use for retrieval.
        env (Environment): The environment instance.
        logger (Logger): Logger instance for logging.
        to_print (bool): Whether to print outputs or not.

    Returns:
        tuple: The final reward (or score) and a dictionary containing task information.
    """
    logger.info("Starting scienceworld_run_rap with task: %s", ob.split('\n')[0])
    
    # Print initial observation if required
    if to_print:
        print(ob)
    ob_prompt = 'Here is the task information.\n' + ob.split('\n')[0] + '\n'
    target_task = ob.split('\n')[0]

    # Planning phase: generate plan using retrieved examples
    examples = generate_examples({'Task': target_task, 'Category': category, 'Plan': ''}, memory, embeddings, k=3, for_plan=True)
    examples_text = 'Here are examples.\n' + ''.join(examples)
    target_prompt = '\nHere is the task. Please make a plan from the examples.\nYour task is to: ' + target_task + '\n> think: To solve the task,'
    full_prompt = examples_text + target_prompt
    plan = llm(full_prompt)
    # Format the plan appropriately
    plan = 'To solve the task, ' + plan.split('.')[0] + '.'
    
    target_prompt = 'Here is the task. Please make an action from the examples.\nTask : ' + target_task + '\nPlan : ' + plan + '\n'
    
    # Prepare task data structure
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
    
    # Initial retrieval of examples for actions
    ret_examples = generate_examples(data, memory, embeddings, k=4, act_len=20)
    
    for i in range(1, args.num_steps):
        # Generate action with retrieval context
        examples = 'Here are examples.\n' + ''.join(ret_examples)
        full_prompt = ob_prompt + examples + target_prompt + '\n'.join(data['Actions']) + '\n>'
        action = llm(full_prompt)
        logger.info("Step %d: Generated action: %s", i, action)
            
        # Execute the action in the environment
        observation, _, done, info = env.step(action)
        reward = info.get('reward', 0)
        fail = (info.get('score', 0) == -100)
        logger.debug("Step %d: Executed action: %s | Observation: %s, Reward: %s, Done: %s, Info: %s",
                     i, action, observation, reward, done, info)
        
        observation = process_ob(observation, env)
        if action.startswith('think:'):
            observation = 'OK.'
        
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            
        # Generate retrieval key based on the action (if applicable)
        if 'think:' in action:
            full_prompt_key = 'Here are examples.\n' + ret_key_examples + '\nHere is the task. Please make a plan from the examples.\n' + action + '\n>'
            retrieve_key = llm(full_prompt_key)
            logger.info("Step %d: Retrieved key: %s", i, retrieve_key)
            if 'search:' in retrieve_key:
                search_object = retrieve_key.split('search:')[1].strip()
                ret_examples = generate_examples(data, memory, embeddings, k=8, act_len=10, mode='obs', key=search_object)
            elif 'action:' in retrieve_key:
                reasoning = retrieve_key.split('action:')[1].strip()
                ret_examples = generate_examples(data, memory, embeddings, k=4, act_len=20, mode='act', key=reasoning)

        # Append the action and observation to the log
        actions.append('> ' + action)
        actions.append(observation)
        data['Actions'] = actions[-10:]
        
        if fail:
            logger.error("Step %d: Action resulted in failure. Exiting with last score: %s", i, last_score)
            return last_score, ''
        
        if done:
            # Remove invalid actions and observations
            import numpy as np
            inv_act_idx = np.where(np.array(actions) == 'Nothing happens.')[0]
            inv_act_idx = np.append(inv_act_idx, inv_act_idx - 1)
            actions = [actions[i] for i in range(len(actions)) if i not in inv_act_idx]
            data['Actions'] = actions
            logger.info("Task completed successfully with final reward: %s", reward)
            return reward, data
        
        last_score = info.get('score', last_score)
    
    logger.info("Reached maximum steps with final score: %s", info.get('score', last_score))
    return info.get('score', last_score), ''

with open('./prompts/rap_prompts/scienceworld_prompt.jsonl', 'r') as f:
    d = json.load(f)    
    
if __name__ == "__main__":
    config = load_config('./baseline_config.ini')
    # with open('./config/config.yml', 'r') as agentconfig:
    #     agent_config = yaml.load(agentconfig, Loader=yaml.FullLoader)
    logger = setup_logger('./rap_logs')
    print('Successfully read!')

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=8, help="The number of trials")
    parser.add_argument("--num_steps", type=int, default=50, help="The number of steps")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The model name")
    parser.add_argument("--output", type=str, default="./rap_logs", help="The output folder")
    parser.add_argument("--emb_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", choices=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"], help="The model name")
    args = parser.parse_args()
    model_embedding = SentenceTransformer(args.emb_model)


    tasks = list(categories.keys())
    simplification = 'easy'

    os.makedirs(args.output, exist_ok=True)

    # with open('./configs/base_config.yaml') as reader:
    #     rap_config = yaml.safe_load(reader)

    ret_key_examples = open('./prompts/rap_prompts/retrieval_prompt.txt').readlines()
    ret_key_examples = ''.join(ret_key_examples)

    rs_trials = []
    env = ScienceWorldEnv()
    memory = []
    
    current_memory = []
    for trial in range(args.num_trials):
        mode = "trainning" if trial < 5 else "evaluating"
        print('### trial '+str(trial+1)+' mode ' +  mode + ' ###')
        # split = "eval_out_of_distribution"
        
        if mode == "evaluating":
            memory = current_memory[:]
            embeddings = generate_embeddings(memory)
    
        tasks = [tasks[15]]
        for task_num in range(len(tasks)):
            task = env.getTaskNames()[task_num]
            category = categories[task]
            env.load(task, variationIdx=trial, simplificationStr=simplification, generateGoldPath=True)
            env.reset()    
            if mode == 'trainning':
                r, mem_data = scienceworld_run_react(categories[task], env, logger)
                if not mem_data=='':
                    current_memory.append(mem_data)
            else:
                r, mem_data = scienceworld_run_rap(env.getTaskDescription(), category, memory, embeddings, env, logger )
                if not mem_data=='':
                    current_memory.append(mem_data)
                # rs_trials.append(r)
                logging.info(f"Task '{task}' variation {trial} evaluated with reward: {r}")

        # # Save memory and scores after each trial
        # memory_file = os.path.join(args.output, f"memory_trial_{trial + 1}.json")
        # scores_file = os.path.join(args.output, "scores.json")
        
        # with open(memory_file, 'w') as f:
        #     json.dump(current_memory, f, indent=4)
        # logging.info(f"Memory saved to {memory_file}")
        
        # with open(scores_file, 'w') as f:
        #     json.dump(rs_trials, f, indent=4)
        # logging.info(f"Scores saved to {scores_file}")

    logging.info("All trials completed.")