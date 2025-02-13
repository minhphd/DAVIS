import logging
import asyncio
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import load_config, get_connection_pool
from scienceworld import ScienceWorldEnv
from ReasoningAgent import ReasoningAgent
from datetime import datetime
import yaml
from concurrent.futures import ProcessPoolExecutor
import sys
import gc

# Function to setup and return a logger
def setup_experiment_logger(log_file, agent_id=None):
    logger_name = f'experiment_logger_{agent_id}' if agent_id else 'experiment_logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Load configuration
config = load_config('./config/config.ini')
with open('./config/config.yml', 'r') as agentconfig:
    agent_config = yaml.load(agentconfig, Loader=yaml.FullLoader)
print('Successfully read!')

# Get seed, hyperparameters, and task from the configuration
seed = int(agent_config['AGENT']['SEED'])
max_look_ahead = agent_config['AGENT']['MAX_LOOK_AHEAD']
max_query = agent_config['AGENT']['MAX_QUERY']

#setting seed
random.seed(seed)
np.random.seed(seed)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the global experiment directory exists
global_experiment_dir = os.path.join(os.path.dirname(__file__), f'log/oracle_{timestamp}')
os.makedirs(global_experiment_dir, exist_ok=True)

# Save configuration file in the global experiment directory
config_save_path = os.path.join(global_experiment_dir, 'config.yml')
with open(config_save_path, 'w') as outfile:
    yaml.dump(agent_config, outfile)

# Setup global logger
global_log_file = os.path.join(global_experiment_dir, 'global_experiment.log')
global_logger = setup_experiment_logger(global_log_file)

global_logger.info(f"Seed: {seed}")
global_logger.info(f'Max look ahead: {max_look_ahead} steps')
global_logger.info(f'Max query: {max_query} queries')

# Initialize the agent parameters
agent_id = agent_config['AGENT']['ID']
agent_model = agent_config['AGENT']['AGENT_MODEL']
agent_kgmodel = agent_config['AGENT']['KNOWLEDGE_GRAPH_MODEL']

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

# Run the experiment for a single agent
def run_agent(agent_config, tasks, variations, simplification, global_experiment_dir):
    # Initialize environment and connection within the process
    try:
        connection_pool = get_connection_pool(config)
        agent_id = agent_config['AGENT']['ID']
        
        #to reduce the cost of training, we use llama3-70b instead of gpt-4-turbo
        agent = ReasoningAgent(config, None, connection_pool, agent_id, 300, max_look_ahead, max_query, ScienceWorldEnv(), 'meta-llama-3-70b-instruct', 'meta-llama-3-70b-instruct')

        for task in tasks:
            # Iterate through multiple variations for each task
            for variation in variations:
                # Generate timestamp    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Ensure the log directory exists and includes timestamp and hyperparameters
                log_dir = os.path.join(global_experiment_dir, f'log/{task}/variation_{variation}_t_{timestamp}/')
                os.makedirs(log_dir, exist_ok=True)

                # Setup log file
                log_file = os.path.join(log_dir, 'experiment.log')
                logger = setup_experiment_logger(log_file, agent_id)
                
                agent.logger = logger
                agent.env.load(task, variationIdx=variation, simplificationStr=simplification, generateGoldPath=True)
                agent.env.reset()
                logger.info(agent.env.getTaskDescription())
                agent.reset()
                
                train(agent)
                                
                global_logger.info(f'finished task: {task} variation: {variation}')
    
    except Exception as e:
        global_logger.error(f"An error occurred in run_agent: {e}", exc_info=True)
        raise e

def train(agent):
    try:
        asyncio.run(agent.update(store=False))
        for action in agent.env.getGoldActionSequence():
            _, res, info, _ = asyncio.run(agent.step(action)) 
            agent.logger.info(f"Executed action `{action}` | received response `{res}` | Total Score {info['score']}`")
            # return 
    except Exception as e:
        agent.logger.error(f"An error occurred in train: {e}", exc_info=True)
        raise e

def run_parallel_experiments(num_agents=2, tasks=[], simplification='easy'):
    ep_per_agent = len(tasks) // num_agents
    task_chunks = list(divide_chunks(tasks, ep_per_agent))
    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = []
        for i in range(num_agents):
            futures.append(executor.submit(run_agent, agent_config, task_chunks[i], [0, 1,2,3,4], simplification, global_experiment_dir))

        results = [f.result() for f in futures]
    return results

# Function to plot scores vs time
def plot_scores_vs_time(scores, t, task, variation, plot_file):
    t_minutes = [(time - min(t)) / 60 for time in t]
    
    plt.figure()
    plt.plot(t_minutes, scores)
    plt.ylim((0, 100))
    plt.xlabel('Time (minutes)')
    plt.ylabel('Scores')
    plt.title(f'{task}_{variation}')
    plt.savefig(plot_file)
    plt.close()

# Function to summarize results
def summarize_results(results, summary_file):
    total_scores = {}
    total_times = {}

    for agent_results in results:
        for task_result in agent_results:
            task = task_result['task']
            if task not in total_scores:
                total_scores[task] = []
                total_times[task] = []
            total_scores[task].extend(task_result['scores'])
            total_times[task].extend(task_result['times'])

    avg_scores = {task: np.mean(scores) for task, scores in total_scores.items()}
    avg_times = {task: np.mean(times) for task, times in total_times.items()}
    
    summary = "\n".join([f"Task: {task} - Average Score: {avg_scores[task]}, Average Time: {avg_times[task]}" for task in avg_scores])
    
    with open(summary_file, 'w') as f:
        f.write(summary)

    print(summary)

# Execute the experiment
if __name__ == '__main__':
    # Experiment parameter
    num_agent = 1
    simplifications = 'easy'
    tasks = ScienceWorldEnv().getTaskNames()
    # agent = ReasoningAgent(config, None, get_connection_pool(config), agent_id, ScienceWorldEnv(), agent_model, agent_kgmodel)
    # agent.reset_memory()
    
    # del agent
    # gc.collect()
    
    try:
        results = run_parallel_experiments(num_agents=num_agent, tasks=tasks, simplification=simplifications)
        
        # summary_file = os.path.join(global_experiment_dir, 'summary.txt')
        # summarize_results(results, summary_file)
        # global_logger.info(f"Summary saved to {summary_file}")

        # global_logger.info(f"Experiment completed with results: {results}")
    except Exception as e:
        global_logger.error(f"An error occurred during the experiment: {e}", exc_info=True)
        sys.exit(1)
