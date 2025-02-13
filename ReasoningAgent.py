"""
Description: 
    This module defines the ReasoningAgent class, which utilizes a knowledge graph 
    and a large language model (LLM) to interact with the ScienceWorld environment.
    The agent performs reasoning tasks, interacts with the environment, updates 
    its internal state, and refines its actions based on feedback.
    
    The module also includes functions for loading configuration, setting up 
    logging, and running experiments.
"""

from utils import *
from kg_graph.KnowledgeGraph import KnowledgeGraph
from scienceworld import ScienceWorldEnv
import time
import gc
import torch
import kg_graph.schema as sm
from pprint import pprint
import os
import asyncio
from threading import Thread
from sentence_transformers import SentenceTransformer

class ReasoningAgent():
    def __init__(self, config, logger, connection_pool, this_uuid, step_limit, env: ScienceWorldEnv, agent_model='gpt-4o', kg_model=''):
        """
        Initializes the ReasoningAgent with the given configuration, logger, connection pool, UUID, and environment.

        Args:
            config (ConfigParser): Configuration object containing various settings.
            logger (logging.Logger): Logger object for logging messages.
            connection_pool (psycopg2.pool.SimpleConnectionPool): Connection pool for database connections.
            this_uuid (str): Unique identifier for the agent instance.
            env (ScienceWorldEnv): Instance of the ScienceWorld environment.
        """
        
        # set up logger
        self.logger = logger
        self.config = config

        # Setup Connection Pool for Agent. Exit if fail.
        self.connection_pool = connection_pool
        # if not self.connection_pool:
        #     print("Failed DB Connection. Exiting")
        #     self.logger.error(f"FAILED STARTUP: Error connecting to database. SHUTDOWN.")
        #     exit()

        self.rel_path = os.path.dirname(__file__)
        self.model = agent_model

        self.kg = KnowledgeGraph(self, prompts_path=self.rel_path+'/kg_graph/sciworld_prompt', config=config, ner_llm=True, model=kg_model)
        self.token_sent = 0
        self.token_received = 0
        self.inventory = ''
        self.env = env
        self.task = ''
        self.total_reflection = 0
        self.observation = ''
        self.past_actions = ['<START>']
        self.action_res = ['NONE']
        self.step_limit = step_limit
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.possible_actions = self.env.getPossibleActions()
        self.possible_objects = self.env.getPossibleObjects()
        self.kg_log = []
        self.this_uuid = this_uuid
        self.trajectory = []
        self.actor_prompt = open(self.rel_path+'/prompts/actor.txt').read()
        self.refiner_prompt = open(self.rel_path+'/prompts/refiner_v2.txt').read()
        self.reflect_prompt = open(self.rel_path+'/prompts/reflection.txt').read()
        self.past_reflections = ""

    def prompts_update(self):
        """
        Updates the prompts for the knowledge graph and the agent.
        """
        self.kg.reload_prompt()
        self.actor_prompt = open(self.rel_path+'/prompts/actor.txt').read()
        self.refiner_prompt = open(self.rel_path+'/prompts/refiner_v2.txt').read()
        self.reflect_prompt = open(self.rel_path+'/prompts/reflection.txt').read()

    def reset(self):
        """
        Resets the internal state of the agent.
        """
        self.inventory = ''
        # self.env = None
        self.task = ''
        self.token_received = 0
        self.token_sent = 0
        self.observation = ''
        self.past_actions = ['<START>']
        self.action_res = ['NONE']
        self.kg_log = []
        self.trajectory = []

        self.kg.reset()

    def reflect(self, text):
        """
        Generates a reflection based on the given text using an LLM. This is for easier parsing into relational database.

        Args:
            text (str): Text to reflect on.

        Returns:
            str: Reflection result from the LLM.
        """
        prompt = self.reflect_prompt.format(text)
        # res = get_octoml_response(self.config, prompt, model='meta-llama-3-70b-instruct')
        res, token_count = get_response(model='meta-llama-3-70b-instruct', prompt=prompt, config=self.config)
        return res, token_count

    async def update(self, action=False, store=True):
        """
        Updates the agent's state by generating new observations and extracting relations.

        Args:
            action (bool): Indicates whether an action has been performed.
            store (bool): Indicates whether to store the extracted relations in the knowledge graph.
        """
        self.observation, token_count = self.reflect(self.env.look())
        self.task = self.env.getTaskDescription()
        self.inventory = self.env.inventory()
        self.possible_actions = self.env.getPossibleActions()
        self.possible_objects = self.env.getPossibleObjects()
        
        #different reflection for action with relocation and without relocation
        if action:
            prompt = f"""
            RECENT ACTION: {self.past_actions[-1:]}
            RESPONSE: {self.action_res[-1:]}
            """
            prompt, token_count = self.reflect(prompt)
        else:
            prompt = self.observation
        if store:
            await self.kg.relations_extraction(prompt, coref_resolve=True, store=store, iterations=1)
            print(f'FINISH SAVING action {self.past_actions[-1]}')
        
    async def step(self, action, store=True):
        """
        Executes a single step in the environment using the given action.

        Args:
            action (str): Action to perform.
            store (bool): Indicates whether to store the extracted relations in the knowledge graph.

        Returns:
            dict: Updated state information.
            str: Next state observation.
            dict: Additional info from the environment.
            bool: Termination status.
        """
        self.past_actions.append(action)
        next_state, _, termination, info = self.env.step(action)
        self.action_res.append(next_state)
        
        #treating relocation actions differently from regular actions
        if "move to" in next_state or "teleport" in next_state:
            await self.update(store=store)
        else:
            await self.update(action = True, store=store)
        
        return {'observation': self.observation, 'inventory': self.inventory, 'valid_receptacles': self.possible_objects}, next_state, info, termination

    async def act_and_refine(self, store=True):
        """
        Performs actions and refines the agent's strategy based on feedback.
        Args:
            max_steps (int): Maximum number of steps to perform in imagination.
            max_query (int): Maximum number of queries to make to the LLM per planning step.
            store (bool): Indicates whether to store the extracted relations in the knowledge graph.

        Returns:
            list: Scores obtained during the execution.
            list: Timestamps of each step.
        """
        reason_model = self.model
        last_token_sent = 0
        last_token_received = 0
        self.trajectory, printout, total_token_sent, total_token_received = self.kg.get_trajectory(
            valid_objects=self.possible_objects,
            task=self.task,
            observation=self.observation,
            inventory=self.inventory,
            MAX_STEPS=self.look_ahead,
            MAX_QUERIES=self.max_query,
            model=reason_model
        )
        self.token_sent += total_token_sent
        self.token_received += total_token_received
        s = self.observation
        print(f'New trajectory: \n{printout}')
        print('<FINISHED INITIAL PLANNING>')
        scores = []
        t = []
        i = 0
        termination = False
        
        #Main training loop, please refer Figure 1 of the submitted PDF
        while i < len(self.trajectory) - 1 and (not termination):
            self.act(i, self.model)
            sar_hat = self.trajectory[i]
            a_hat = findValidActionNew(sar_hat['action'], self.env, self.observation, self.past_actions, self.sbert, self.logger)
            
            try:
                s_next, r, info, termination = await self.step(a_hat, store)
                print(info['moves'], termination)
            except KeyboardInterrupt:
                print("Manual interruption detected. Stopping...")
                return  # Exit the method upon interruption
            
            scores.append(info['score'])
            t.append(time.time())
            print(f't = {time.time()}: Exectued action `{self.past_actions[-1:]}` | received response `{self.action_res[-1:]}` | number of token sent so far: {self.token_sent} | number of token received so far: {self.token_received}')
            self.logger.info(f"t = {time.time()}: Exectued action `{self.past_actions[-1:][0]}` | received response `{self.action_res[-1:][0]} | Total Score {info['score']} | number of token sent so far: {self.token_sent} | number of token received so far: {self.token_received}`")
            if termination:
                break    
            
            if len(self.past_actions) - 1 >= self.step_limit:
                self.logger.info(f"Reached step limit of {self.step_limit}. Finished with a score of {max(scores)}")
                break
            
            sar = {
                'state': s,
                'action': a_hat,
                'env response': r,
                'next state': s_next
            }
            
            # Check whether need to replan
            prompt = self.refiner_prompt.format(self.past_reflections, self.task, info['score'], sar_hat, sar)
            res, token_count = retry(5, get_response, sm.Replanning, reason_model, prompt, json=True, config=self.config)
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            
            if res['should_replan'] or (i == len(self.trajectory) - 2 and not termination) :
                self.past_reflections = res['reflection']
                print('<EXCEPTTION OCCURED, REPLANNING>')
                pprint(res)
                
                # Preserve the initial part of the trajectory before the replan point
                self.trajectory = self.trajectory[:i] + [sar]
                
                new_subtask = f"""
                main_task: {self.task}
                    mini_subtask: {res['updated_subtask']}
                """
                
                # Generate the remaining part of the trajectory after the replan point
                new_trajectory, printout, total_token_sent, total_token_received = self.kg.get_trajectory(
                    valid_objects=self.possible_objects,
                    task=new_subtask,
                    observation=self.observation,
                    inventory=self.inventory,
                    sequence=self.trajectory,
                    reflection=self.past_reflections,
                    MAX_STEPS=self.look_ahead,
                    MAX_QUERIES=self.max_query,
                    model=reason_model
                )
                self.token_sent += total_token_sent - last_token_sent
                self.token_received += total_token_received - last_token_received
                
                last_token_sent = total_token_sent
                last_token_received = total_token_received
                
                # Concatenate the preserved initial part with the new remaining part
                self.trajectory = new_trajectory
                print(f'New trajectory: \n{printout}')
                print('<FINISHED REPLANNING - RESUMING>')
            else:
                self.trajectory[i] = sar
            i += 1
            s = s_next
            
        print('FINISHED RUN!')
        return scores, t

    #Critic
    def refine(self, reason_model, sar_hat, sar):
        prompt = self.refiner_prompt.format(self.task, sar['env response'], sar['score'], sar_hat, sar)
        res = retry(5, get_response, sm.Replanning, reason_model, prompt, json=True, config=self.config)
        return res
    
    #Actor
    def act(self, pointer, model):
        """
        Actor method. This functions call an LLM, given a goal, to translate that goal into actionable commands within the environment. The output will be in the format of a json object where `actions` are translated actions and   `responses` are env response from corresponding action.
        """
        sar_hat = self.trajectory.pop(pointer)
        try: 
            prompt = self.actor_prompt.format(sar_hat['action'], self.observation, self.inventory, self.env.getPossibleObjects())
            res, token_count = retry(5, get_response, sm.Actor, model, prompt, json=True, config=self.config)
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            
            cur_s = sar_hat['state']
            for i in range(len(res['actions'])-1, -1, -1):
                sar = {
                    'state': cur_s,
                    'action': res['actions'][i],
                    'reward (env response)': res['responses'][i],
                    'next state': res['next_states'][i],
                    'done or termination': False
                }
                cur_s = res['next_states'][i]
                if i == len(res['actions']) - 1:
                    sar['next state'] = sar_hat['next state']
                    sar['done or termination'] = sar_hat['done or termination']
                self.trajectory.insert(pointer, sar)
                
            if res['actions'] == []:
                raise(Exception)
            print(f'replaced action {[sar_hat["action"]]} with {res["actions"]}')
            return True
        except Exception as e:
            self.logger.error(f'ERROR ACTING: {e}')
            print(f'Exception: {e} \n {res}')
            self.trajectory.insert(pointer, sar_hat)
            return False

    def reset_memory(self):
        """
        Removes entries from the fact_tuples table where agent_uuid matches self.uuid.
        Also removes any orphaned entities and relationships that are no longer referenced.
        """
        conn = None
        cur = None
        try:
            # Establish database connection and cursor
            conn = self.connection_pool.getconn()
            cur = conn.cursor()

            # Delete records from fact_tuples where agent_uuid matches self.uuid
            cur.execute("DELETE FROM fact_tuples WHERE agent_uuid = %s RETURNING source_entity_id, target_entity_id, relationship_id;", (self.this_uuid,))

            # Collect the ids of affected entities and relationships
            affected = cur.fetchall()
            entity_ids = {row[0] for row in affected}.union({row[2] for row in affected})
            relationship_ids = {row[1] for row in affected}

            # Commit the deletions in fact_tuples
            conn.commit()

            # Now remove any orphaned entities and relationships
            if entity_ids:
                placeholders = ', '.join(['%s'] * len(entity_ids))
                cur.execute(f"DELETE FROM entities WHERE id NOT IN (SELECT DISTINCT source_entity_id FROM fact_tuples) AND id NOT IN (SELECT DISTINCT target_entity_id FROM fact_tuples) AND id IN ({placeholders});", tuple(entity_ids))

            if relationship_ids:
                placeholders = ', '.join(['%s'] * len(relationship_ids))
                cur.execute(f"DELETE FROM relationships WHERE id NOT IN (SELECT DISTINCT relationship_id FROM fact_tuples) AND id IN ({placeholders});", tuple(relationship_ids))

            # Commit the final deletions
            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Failed to reset memory for uuid {self.uuid}. Error: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)

    async def interactive_mode(self, store=False):
        """
        Allows for direct human interaction with the agent in a while loop for data collection purposes.
        """
        print("Entering interactive mode. Type 'exit' to quit.")
        try:
            while True:
                human_action = input("Enter action: ").strip()
                
                if human_action.lower() == 'exit':
                    print("Exiting interactive mode.")
                    break
                
                # Simulate the human action in the environment and update the agent's state
                response, _, termination, info = self.env.step(human_action)
                print(f"Environment Response: {response}")
                
                # Update agent's internal state
                if "move to" in response or "teleport" in response:
                    # print('new')
                    await self.update(store=store)
                else:
                    await self.update(action = True, store=store)
                
                if termination:
                    print("The task has been terminated based on the action taken.")
                    break
                
                
        except KeyboardInterrupt:
            print("Interactive mode interrupted.")

    def get_uuid(self):
        return self.this_uuid