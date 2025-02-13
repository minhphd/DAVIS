import logging
import asyncio
import os
import random
import pickle
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from utils import load_config, get_connection_pool
from scienceworld import ScienceWorldEnv
from ReasoningAgent import ReasoningAgent
from datetime import datetime
import yaml
from concurrent.futures import ProcessPoolExecutor
from utils import *

# Function to setup and return a logger
def setup_experiment_logger(log_file, agent_id=None):
    logger_name = f'experiment_logger_{agent_id}' if agent_id else 'experiment_logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
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

random.seed(seed)
np.random.seed(seed)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the global experiment directory exists
global_experiment_dir = os.path.join(os.path.dirname(__file__), f'log/experiment_{timestamp}')
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

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def run_agent(agent_config, tasks, set, simplification, global_experiment_dir, max_steps):
    # Initialize environment and connection within the process
    try:
        connection_pool = get_connection_pool(config)
        agent_id = agent_config['AGENT']['ID']
        agent_model = agent_config['AGENT']['AGENT_MODEL']
        agent_kgmodel = agent_config['AGENT']['KNOWLEDGE_GRAPH_MODEL']
        
        agent = ReasoningAgent(config, None, connection_pool, agent_id, max_steps, max_look_ahead, max_query, ScienceWorldEnv(), agent_model, agent_kgmodel)
        all_scores = []
        all_times = []
        for task in tasks:
            agent.env.load(task, variationIdx=0, simplificationStr=simplification, generateGoldPath=True)
            variations = load_variation(agent.env, set)
            for variation in variations:
                # Generate timestamp    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Ensure the log directory exists and includes timestamp and hyperparameters
                log_dir = os.path.join(global_experiment_dir, f'log/{task}/var_{variation}_t_{timestamp}/')
                os.makedirs(log_dir, exist_ok=True)

                # Setup log file
                log_file = os.path.join(log_dir, 'experiment.log')
                logger = setup_experiment_logger(log_file, agent_id)
                
                agent.logger = logger
                agent.env.load(task, variationIdx=variation, simplificationStr=simplification, generateGoldPath=True)
                agent.env.reset()
                logger.info(agent.env.getTaskDescription())
                agent.reset()
                
                scores, t = asyncio.run(run_single_experiment(agent))
                all_scores.append(scores)
                all_times.append(t)
                global_logger.info(f'Finished variation {variation} of task {task} with a score of {max(scores)}')
                plot_scores_vs_time(scores, t, task, variation, os.path.join(log_dir, 'figure.png'))
        
        return {'task': task, 'scores': all_scores, 'times': all_times}
    
    except Exception as e:
        global_logger.error(f"An error occurred: {e}")
        raise e

async def run_single_experiment(agent):
    await agent.update(store=False)
    scores, t = await agent.act_and_refine(store=False)
    agent.logger.info(f"Scores: {scores}, Time: {t}")
    return scores, t

def run_parallel_experiments(num_agents=2, tasks=[], variations=[], simplification='easy', max_steps=40, set='test_mini'):
    task_chunks = list(divide_chunks(tasks, len(tasks) // num_agents))
    
    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = []
        for i in range(num_agents):
            futures.append(executor.submit(run_agent, agent_config, task_chunks[i], set, simplification, global_experiment_dir, max_steps))

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

    print(results)
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
    # Experiment parameters
    num_agent = agent_config['EXPERIMENT']['NUM_AGENTS']
    num_eps = agent_config['EXPERIMENT']['NUM_EPS']
    set = agent_config['EXPERIMENT']['SET']
    starting_var = 5
    variations = list(range(starting_var, starting_var + num_eps, 1))
    max_steps = agent_config['EXPERIMENT']['MAX_STEPS']
    simplifications = agent_config['EXPERIMENT']['SIMPLIFICATION']
    tasks = agent_config['EXPERIMENT']['TASKS']
    results = run_parallel_experiments(num_agent, tasks, variations, simplifications, max_steps, set)
    
    summary_file = os.path.join(global_experiment_dir, 'summary.txt')
    summarize_results(results, summary_file)
    global_logger.info(f"Summary saved to {summary_file}")

    global_logger.info(f"Experiment completed with results: {results}")
