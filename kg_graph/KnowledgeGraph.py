"""
Description: 
    A class for constructing knowledge graphs from text using NLP and LLM models.

    This class leverages Stanford CoreNLP for coreference resolution and named entity recognition, 
    while utilizing large language models (LLM) for relation extraction and entity linking. 
    Coreference resolution helps resolve pronouns and referring expressions to their corresponding entities. 
    Entities are extracted with Stanford CoreNLP's named entity recognition, and relationships between entities 
    are inferred using custom LLM prompts. The class optionally performs coreference resolution before entity 
    and relation extraction to improve knowledge graph construction accuracy.
"""


import stanza
stanza.download('en')
from utils import *
from pyvis.network import Network
from openai import AsyncOpenAI
import asyncio
from collections import defaultdict
from psycopg2.extras import RealDictCursor
from collections import deque 


import re
import sys
import ast
import kg_graph.schema as sm
from time import time
from pprint import pprint

class KnowledgeGraph:
    def __init__(self, 
                 agent,
                 prompts_path='kg_graph/conv_prompt',
                 config=None, 
                 model='gpt-4o',
                 ner_llm=True):
        """
        A KnowledgeGraph object is responsible for constructing a knowledge graph from textual data, utilizing NLP and LLM models.
        
        Attributes:
            agent (CompHuSimAgent): The agent owning this KnowledgeGraph instance.
            prompts_path (str): Path to the directory containing prompts for LLM queries.
            config (dict): Configuration settings for the LLM model.
            model (str): Identifier for the LLM model used for relation extraction and other NLP tasks.
            ner (Stanza Pipeline object, optional): NLP pipeline for Named Entity Recognition. Defaults to None.
            coref (Stanza Pipeline object): NLP pipeline for Coreference Resolution.
        """
        
        #Setup Connection Pool for Agent. Exit if fail.
        self.connection_pool = agent.connection_pool
        self.logger = agent.logger

        if not self.connection_pool:
            print("Failed DB Connection. Exiting")
            self.logger.error(f"FAILED STARTUP: Error connecting to database. SHUTDOWN.")
            exit()

        if not agent: 
            print('KnowledgeGraph has to belong to an agent, please provide CompHuSimAgent')
            exit()
        
        self.conn = self.connection_pool.getconn()
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        #owner
        self.agent = agent
        
        #NLP model
        self.ner = None
        if not ner_llm:
            self.ner = stanza.Pipeline('en', processors='tokenize, ner')
        self.coref = stanza.Pipeline('en', processors='tokenize, coref')
        self.lemma = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

        #prompt groups
        self.token_sent = 0
        self.token_received = 0
        self.octo_token = 0
        self.data_summary_prompt = open(f'{prompts_path}/data_summary_prompt.txt').read()
        self.relational_prompt = open(f'{prompts_path}/prompt_edge_infer.txt').read()
        self.entities_extract_prompt = open(f'{prompts_path}/entities_extraction_prompt.txt').read()
        self.query_prompt = open(f'{prompts_path}/query_prompt.txt').read()
        self.types_search = open(f'{prompts_path}/types_search.txt').read()
        self.action_gen = open(f'{prompts_path}/action_generator.txt').read()
        self.state_gen = open(f'{prompts_path}/next_state_generator.txt').read()
        self.prompts_path = prompts_path
        
        
        #kg_log
        self.kg_log = ""
        
        self.config = config
        self.model = model

    def reload_prompt(self):
        """
        Reloads the text prompts from the specified directory path into the KnowledgeGraph instance.
        
        This method ensures that any updates to the prompt files are reflected in the KnowledgeGraph operations.
        """
        self.data_summary_prompt = open(f'{self.prompts_path}/data_summary_prompt.txt').read()
        self.relational_prompt = open(f'{self.prompts_path}/prompt_edge_infer.txt').read()
        self.entities_extract_prompt = open(f'{self.prompts_path}/entities_extraction_prompt.txt').read()
        self.query_prompt = open(f'{self.prompts_path}/query_prompt.txt').read()
        self.types_search = open(f'{self.prompts_path}/types_search.txt').read()
        self.action_gen = open(f'{self.prompts_path}/action_generator.txt').read()
        self.state_gen = open(f'{self.prompts_path}/next_state_generator.txt').read()

    def reset(self):
        self.token_sent = 0
        self.token_received = 0
        self.octo_token = 0
        self.reload_prompt()
        self.connection_pool = self.agent.connection_pool
        self.logger = self.agent.logger

        self.kg_log = ""


    def coref_resolve(self, text):
        """
        Performs coreference resolution on the given text, replacing pronouns and referring expressions with the entities they refer to.
        
        Parameters:
            text (str): The text for which coreference resolution is to be performed.
        
        Returns:
            str: The text with resolved coreferences.
        """
        doc = self.coref(text)
        resolved_doc = ""
        for sentence in doc.sentences:
            is_resolving = False
            for token in sentence.words:
                if len(token.coref_chains) != 0 or is_resolving:
                    longest_representative = None
                    for chain in token.coref_chains:
                        current_representative = chain.to_json()
                        if longest_representative is None or len(current_representative['representative_text']) > len(longest_representative['representative_text']):
                            longest_representative = current_representative
                    representative = longest_representative
                    
                    if representative:
                        if 'is_start' in representative and representative['is_start']:
                            is_resolving = True
                            resolved_doc += " " + representative['representative_text']
                        if 'is_end' in representative and representative['is_end']:
                            is_resolving = False
                        
                elif not is_resolving:
                    resolved_doc += " " + token.text
        return resolved_doc


    def lemmatize(self, text):
        """
        Performs lemmatization on the given text, converting words to their base forms.

        Parameters:
            text (str): The text to be lemmatized.

        Returns:
            str: The lemmatized text.
        """
        doc = self.lemma(text)
        lemmatized_text = " ".join([word.lemma for sent in doc.sentences for word in sent.words])
        return lemmatized_text
    

    def extract_entities(self, text, coref_resolve=False, model='meta-llama-3-70b-instruct'):
        """
        Extracts named entities from the given text, optionally performing coreference resolution first.
        
        Parameters:
            text (str): The text from which entities are to be extracted.
            coref_resolve (bool): Whether to perform coreference resolution before entity extraction. Defaults to False.
        
        Returns:
            list: A list of extracted entities.
            str: The processed text, potentially with resolved coreferences.
        """
        if not coref_resolve:
            text = self.coref_resolve(text)
        text = self.lemmatize(text)
        
        try:
            if not self.ner:
                prompt = self.entities_extract_prompt.format(text)
                entities, token_count = retry(10, get_response, sm.Entities, model, prompt, json=True, config=self.config)
                # entities = get_response(model, prompt, json=True, config=self.config)
            else: 
                entities = self.ner(text).ents
            return entities, text
        except Exception as e:
            return {}, text

    async def relations_extraction(self, text, entities=None, coref_resolve=False, store=True, iterations=1):
        """
        Extracts relations between entities from the text using LLM, potentially across multiple iterations for enhanced detail.
        
        Parameters:
            text (str): The text from which relations are to be extracted.
            entities (list, optional): A list of entities to consider for relation extraction. Extracted from text if not provided.
            coref_resolve (bool): Whether to perform coreference resolution before relation extraction. Defaults to False.
            store (bool): Whether to store extracted relations in the database. Defaults to True.
            json (bool): Whether to return the relations as JSON objects. Defaults to False.
            iterations (int): Number of iterations for relation extraction to capture more details. Defaults to 3.
        
        Returns:
            list: A list of extracted relations in the format (entity1, relation, entity2).
            str: The processed text, potentially with resolved coreferences.
        """
        try: 
            # start_time = time()
            agent_uuid = self.agent.get_uuid()
            
            best_triples = defaultdict(lambda: ('', '', ''), {})
            
            if coref_resolve:
                text = self.coref_resolve(text)

            # print(f'finished coref resolve in : {time() - start_time}')
            # print('resolved text: ', text)
            # start_time = time()
            
            
            if not entities:
                entities, text = self.extract_entities(text, coref_resolve=True, model=self.model)

            # print(entities)
            # print(f'finished NER in {time() - start_time}')
            # print(entities)
            # start_time = time()

            tasks = []
            prompt = self.relational_prompt.format(text, entities)
            if iterations == 1:
                responses = [retry(10, get_response, sm.Relations, self.model, prompt, json=True, config=self.config)]
                # responses = [get_response(self.model, prompt, json=True, config=self.config)]
            else:
                for _ in range(iterations):
                    tasks.append(self.get_async_gpt_response(prompt, model=self.model, response_type='json_object'))
                responses = await asyncio.gather(*tasks)
            
            # print(f'finshed rel extraction in {time() - start_time}')
            # print(responses[0][0])
            # start_time = time()
            
            for res in responses:
                # print(res)
                res, token_count = res
                self.token_sent += token_count['sent']
                self.token_received += token_count['received']
                triples = res['rels']
                
                for triple in triples:
                    if len(triple) == 3:
                        #only process valid trips
                        key = (triple[0], triple[2])  # (entity1, entity2) as the key
                        if len(triple[1]) > len(best_triples[key][1]):  # Check if the new relation is more detailed
                            best_triples[key] = triple
            
            all_triples = list(best_triples.values())

            # print(f'filter and cleaning rels in {time() - start_time}')
            # print(all_triples)
            # start_time = time()

            if store:
                self.store_knowledge_graph(all_triples, entities, agent_uuid)


            # print(f'finshed storing in {time() - start_time}')
            # start_time = time()
            
            return res, text
        except Exception as e:
            self.logger.error('ERROR SAVING: ', e)
            print(f'error saving {e}')
            return

    async def get_async_gpt_response(self, gpt_prompt, model='gpt-3.5-turbo',response_type='text'):
        """
        Use OpenAI's API to generate a response to the given prompt.

        Parameters:
            config (ConfigParser): The configuration parser object that contains the API key.
            gpt_prompt (str): The prompt to pass to API
            model (str): The MODEL to use: SEE: https://platform.openai.com/docs/models

        Returns:
            str: The content of the response generated by API

        Note:
            - The temperature parameter controls the randomness of the output. A higher value makes the output more random, while a lower value makes it more deterministic.
            - The max_tokens parameter controls the maximum length of the generated response.
            - The frequency_penalty parameter can be used to reduce the likelihood of frequent words/phrases.
        """
        api_key = self.config.get('DEFAULT', 'LLM_API_KEY')
        client = AsyncOpenAI(api_key=api_key)


        messages = [{"role": "system", "content": gpt_prompt}]

        response_type_object = {"type": response_type}
        
        args = {
            "model": model,
            "temperature": 0.3,  # description of temperature: http://bit.ly/3rmAJqu
            "max_tokens": 2000,
            "frequency_penalty": 0.0
        }
        
        
        if 'instruct' in model:
            args["prompt"] = gpt_prompt
            response = await client.completions.create(**args)
            content = response.choices[0].text #The legacy way (as of NOv 8, 2023)
        else:
            if response_type == 'json_object':
                args["response_format"] = response_type_object
            args["messages"] = messages
            response = await client.chat.completions.create(**args)
            content = response.choices[0].message.content

        if response_type == 'json_object':
            content = json_parser(self.config, content)

        #return json_objects
        return content

    def store_knowledge_graph(self, fact_tuples, ner_res, agent_uuid=None):
        """
        Stores the extracted knowledge graph information in the database, ensuring that entities are only stored
        if they are part of relations that are successfully inserted into the database.
        
        Parameters:
            fact_tuples (list of tuples): Extracted relations to be stored.
            ner_res (list): Named entity recognition results.
            agent_uuid (str, optional): UUID of the agent owning the knowledge graph. Uses the instance's agent if not provided.
        """
        try: 
            if not agent_uuid:
                agent_uuid = self.agent.get_uuid()
            
            # Set up database connection and cursor
            with self.connection_pool.getconn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                entities = {e[0]: e[1] for e in ner_res['entities']}
                entity_ids = {}
                relation_ids = {}

                # First check and insert relations, then process entities if relations exist
                new_facts = []
                for subject, relation, obj in fact_tuples:
                    if subject in entities and obj in entities:
                        new_facts.append((subject, relation, obj))
                        
                # Insert relations and check existing ones
                for _, relation, _ in new_facts:
                    if relation not in relation_ids:
                        relation_ids[relation] = self._insert_relation(cur, relation)

                # Process entities only if their relations are being inserted
                for subject, relation, obj in new_facts:
                    if relation_ids[relation]:  # Only process entities if the relation was processed
                        if subject not in entity_ids:
                            entity_ids[subject] = self._insert_entity(cur, subject, entities[subject])
                        if obj not in entity_ids:
                            entity_ids[obj] = self._insert_entity(cur, obj, entities[obj])

                        # Prepare for insertion into fact_tuples
                        subject_id = entity_ids[subject]
                        relation_id = relation_ids[relation]
                        object_id = entity_ids[obj]
                        self._insert_fact_tuple(cur, subject_id, relation_id, object_id, agent_uuid)
            self.connection_pool.putconn(conn)
            cur.close()
            
        except Exception as e:
            self.logger.error('ERROR STORING: ', e)
            print(f"Error storing knowledge graph: {e}")

    def _insert_entity(self, cur, name, etype):
        """
        Inserts an entity into the database if it doesn't already exist.
        """
        cur.execute("SELECT id FROM entities WHERE name = %s;", (name,))
        result = cur.fetchone()
        if result:
            return result['id']
        else:
            cur.execute("INSERT INTO entities (name, type) VALUES (%s, %s) RETURNING id;", (name, etype,))
            return cur.fetchone()['id']

    def _insert_relation(self, cur, name):
        """
        Inserts a relation into the database if it doesn't already exist.
        """
        cur.execute("SELECT id FROM relationships WHERE relationship_type = %s;", (name,))
        result = cur.fetchone()
        if result:
            return result['id']
        else:
            cur.execute("INSERT INTO relationships (relationship_type) VALUES (%s) RETURNING id;", (name,))
            return cur.fetchone()['id']

    def _insert_fact_tuple(self, cur, source_id, relation_id, target_id, agent_uuid):
        """
        Inserts a fact tuple into the database if it doesn't already exist.
        """
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM fact_tuples
                WHERE source_entity_id = %s AND relationship_id = %s AND target_entity_id = %s AND agent_uuid = %s
            );
        """, (source_id, relation_id, target_id, agent_uuid))
        exists = cur.fetchone()['exists']
        if not exists:
            cur.execute("""
                INSERT INTO fact_tuples (source_entity_id, relationship_id, target_entity_id, agent_uuid) 
                VALUES (%s, %s, %s, %s);
            """, (source_id, relation_id, target_id, agent_uuid))

    def get_all_entities(self, type=None):
        """
        Retrieves all entities from the database that are part of a fact tuple for the current agent, 
        optionally filtered by type.
        
        Parameters:
            type (tuple, optional): Types of entities to retrieve. Defaults to None.
        
        Returns:
            dict: A mapping of entity names to their IDs.
        """
        cur = None
        conn = None
        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            base_query = """
            SELECT e.id, e.name 
            FROM entities e
            JOIN fact_tuples ft ON e.id = ft.source_entity_id OR e.id = ft.target_entity_id
            WHERE ft.agent_uuid = %s
            """

            if type:
                cur.execute(base_query + "AND e.type IN %s;", (self.agent.get_uuid(), type))
            else:
                cur.execute(base_query, (self.agent.get_uuid(),))

            entities = cur.fetchall()
            return {entity['name']: entity['id'] for entity in entities}
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                self.connection_pool.putconn(conn)


    def get_related_relations(self, entity_ids, agent_uuid=None):
        """
        Retrieves relations related to the specified entities from the database.
        
        Parameters:
            entity_ids (list of int): IDs of entities whose relations are to be retrieved.
            agent_uuid (str, optional): UUID of the agent owning the knowledge graph. Defaults to None.
        
        Returns:
            list: A list of dictionaries representing the related relations.
        """
        # Initialize variables for connection and cursor to None
        conn = None
        cur = None

        try:
            # Ensure connection is set up
            conn = self.connection_pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if not agent_uuid:
                agent_uuid = self.agent.get_uuid()
            
            # Convert the list of entity IDs to a tuple for SQL query
            entity_ids_tuple = tuple(entity_ids)
            
            # SQL query
            query =  """
                SELECT ft.*, e1.name AS source_entity_name, e2.name AS target_entity_name, r.relationship_type
                FROM fact_tuples ft
                JOIN entities e1 ON ft.source_entity_id = e1.id
                JOIN relationships r ON ft.relationship_id = r.id
                JOIN entities e2 ON ft.target_entity_id = e2.id
                WHERE agent_uuid = %s
                AND (source_entity_id IN %s OR target_entity_id IN %s);
                """
            
            # Execute the query with parameters
            cur.execute(query, (agent_uuid, entity_ids_tuple, entity_ids_tuple))
            
            # Fetch and return the results
            return cur.fetchall()
        finally:
            # Ensure resources are always cleaned up properly
            if cur is not None:
                cur.close()
            if conn is not None:
                self.connection_pool.putconn(conn)    

    def bfs_path(self, start, end):
        """
        Finds a path between two entities using Breadth-First Search (BFS).
        
        Parameters:
            start (int): The starting entity ID.
            end (int): The target entity ID.
        
        Returns:
            list: The path of entity IDs from start to end, if exists.
        """
        agent_uuid = self.agent.get_uuid()
        conn = self.connection_pool.getconn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT ft.target_entity_id AS neighbor_id
            FROM fact_tuples ft
            WHERE ft.agent_uuid = %s AND ft.source_entity_id = %s
            UNION
            SELECT ft.source_entity_id AS neighbor_id
            FROM fact_tuples ft
            WHERE ft.agent_uuid = %s AND ft.target_entity_id = %s;
            """
        
        visited = {start: None}  # Maps each node to its predecessor in the path
        queue = deque([start])
        try: 
            while queue:
                current = queue.pop()
                if current == end:
                    break  # Exit if the end entity is found

                # Fetch neighbors from the graph
                cur.execute(query, (agent_uuid, current, agent_uuid, current))
                neighbors = [row['neighbor_id'] for row in cur.fetchall()]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = current  # Map the neighbor back to the current node
                        queue.append(neighbor)

            self.connection_pool.putconn(conn)
            cur.close()
            
            # Reconstruct the path from end to start (if exists)
            path = []
            while end is not None:
                path.append(end)
                end = visited[end]
            return path[::-1]  # Return reversed path
        except Exception as e:
            return [start,end]

    def get_k_hop_neighbors(self, path_entities, agent_uuid, k):
        """
        Retrieves k-hop neighbors for entities along a path.
        
        Parameters:
            path_entities (list of int): Entity IDs forming a path.
            agent_uuid (str): UUID of the agent owning the knowledge graph.
            k (int): The number of hops to consider.
        
        Returns:
            list: A list of relations within k hops of the path entities.
        """
        conn = self.connection_pool.getconn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        all_relations = {}
        entities_to_explore = set(path_entities)
        
        for _ in range(k):
            if not entities_to_explore:
                break
            
            entities_tuple = tuple(entities_to_explore)
            query_relations = """
                SELECT ft.*, e1.name AS source_entity_name, e2.name AS target_entity_name, 
                r.relationship_type, ft.created_at
                FROM fact_tuples ft
                JOIN entities e1 ON ft.source_entity_id = e1.id
                JOIN relationships r ON ft.relationship_id = r.id
                JOIN entities e2 ON ft.target_entity_id = e2.id
                WHERE ft.agent_uuid = %s
                AND (ft.source_entity_id IN %s OR ft.target_entity_id IN %s);
                """
            
            cur.execute(query_relations, (agent_uuid, entities_tuple, entities_tuple))
            results = cur.fetchall()
            for trip in results:
                all_relations[trip['id']] = trip
            
            # Update entities to explore for the next hop
            new_entities = {r['source_entity_id'] for r in results}.union({r['target_entity_id'] for r in results})
            entities_to_explore = new_entities - set(path_entities)
            path_entities.extend(entities_to_explore)
        
        self.connection_pool.putconn(conn)
        cur.close()
        return list(all_relations.values())

    def get_temporal_k_hop(self, entity_ids, agent_uuid=None, k=3):
        """
        Retrieves a temporally ordered list of k-hop neighbors for specified entities.
        
        Parameters:
            entity_ids (list of int): The IDs of entities to explore.
            agent_uuid (str, optional): UUID of the agent owning the knowledge graph. Defaults to None.
            k (int): The number of hops to consider. Defaults to 3.
        
        Returns:
            list: A list of relations within k hops of the specified entities, ordered by time.
        """
        if not agent_uuid:
            agent_uuid = self.agent.get_uuid()
        
        if len(entity_ids) < 2:
            return []  # Need at least two entities to define a path
        
        # Find a static path between the two entities
        path_entities = self.bfs_path(*entity_ids)
        # print(path_entities)
        # Get k-hop neighbors from the path
        all_relations = self.get_k_hop_neighbors(path_entities, agent_uuid, k)
        
        # Sort all relations by time in increasing order
        all_relations.sort(key=lambda x: x['created_at'])
        
        return all_relations

    def _validate_entity_types(self, output):
        """
        Validates the selected entity types based on the query and given instructions.

        Parameters:
        - query (str): The query string.
        - output (dict): The selected entity types in the format {index: "type"}.

        Returns:
        - bool: True if the output is valid, False otherwise.
        """
        # Define the allowed entity types
        allowed_types = ["OBJ", "LOC", "PER"]

        # Ensure output is a dictionary with integer keys and string values
        if not isinstance(output, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in output.items()):
            print("Output should be a dictionary with integer keys and string values.")
            return False

        # Ensure all values are among the allowed entity types
        if not all(v in allowed_types for v in output.values()):
            print("Output contains invalid entity types.")
            return False

        # If all checks pass, the output is valid
        return True

    def query(self, query, additional_context="", asker_uuid=None, summarize=False):
        """
        Queries the knowledge graph and optionally summarizes the results.
        
        Parameters:
            query (str): The query string.
            asker_uuid (str, optional): UUID of the agent making the query. Defaults to None.
            summarize (bool): Whether to summarize the query results. Defaults to False.
        
        Returns:
            The query results, potentially summarized.
        """
        try:
            if not asker_uuid:
                asker_uuid = self.agent.get_uuid()
            
            all_types = ['OBJ', 'LOC', 'PER']

            
            #get related types to filter out entity list
            # types = get_response(self.model, self.types_search.format(query), json=True, config=self.config)
            # print(types)
            types, token_count = retry(5, get_response, self._validate_entity_types , self.model, self.types_search.format(query), json=True, config=self.config)
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']

            types_tuple = tuple([all_types[int(i)] for i in types.keys()])
            filtered_entities = self.get_all_entities(types_tuple)
            
            # results = retry(10, get_response, lambda a: isinstance((a), list), self.model, self.query_prompt.format(query, filtered_entities), config=self.config)
            
            results, token_count = get_response(self.model, self.query_prompt.format(query, filtered_entities), config=self.config)
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            
            # print(results)
            self.octo_token += token_count['sent'] + token_count['received']
            
            # print(results)
            match = re.search(r"\[(.*?)\]", results)
            
            if match:
                results = f"[{match.group(1)}]"
            try:
                entities_list = ast.literal_eval(results)
                related_entites = self.get_temporal_k_hop([filtered_entities[entity] for entity in entities_list], k=2)
            except Exception as e:
                print(e)
                related_entites = []
            
            format = '%H:%M:%S'
            timestamps = ""
            for trip in related_entites:
                t = trip['created_at'].strftime(format)
                timestamps += f'{t}: {trip["source_entity_name"]} -- {trip["relationship_type"]} --> {trip["target_entity_name"]}\n'
            
            # print(timestamps)
            
            # return timestamps
            
            res, token_count = get_response('gpt-4o', self.data_summary_prompt.format(additional_context + "\n GRAPH: \n" + timestamps, query), config=self.config)
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            
            self.octo_token += token_count['sent'] + token_count['received']
            
            return res 

        except Exception as e:
            self.logger.error('ERROR QUERYING: ', e)
            print(f'error querying {e}')
        
    def visualize(self, relations):
        """
        Visualizes the extracted relations as a knowledge graph.
        
        Parameters:
            relations (list of tuples): The relations to visualize.
        """
        # Create a PyVis network with directed edges
        net = Network(notebook=True, directed=True)

        added_entities = set()
        
        # Add nodes and edges
        for source, relation, target in relations:
            if source not in added_entities:
                net.add_node(source, label=source)
                added_entities.add(source)
            if target not in added_entities:
                net.add_node(target, label=target)
                added_entities.add(target)
                
            # Add directed edge with a title
            net.add_edge(source, target, title=relation, arrowStrikethrough=True)

        # Generate and display the graph
        net.show_buttons(filter_=True)
        net.show("network.html")
             
    def visualize_all(self, agent_uuid=None):
        # Ensure connection and cursor are set up
        conn = None
        cur = None
        try: 
            if not conn:
                conn = self.connection_pool.getconn()
            if not cur:
                cur = conn.cursor(cursor_factory=RealDictCursor)
            if not agent_uuid:
                agent_uuid = self.agent.get_uuid()
            
            # Include JOINs to fetch names instead of IDs
            cur.execute("""
                SELECT e1.name AS source_entity_name, r.relationship_type, e2.name AS target_entity_name 
                FROM fact_tuples ft
                JOIN entities e1 ON ft.source_entity_id = e1.id
                JOIN entities e2 ON ft.target_entity_id = e2.id
                JOIN relationships r ON ft.relationship_id = r.id
                WHERE agent_uuid = %s;
                """, (agent_uuid,))
            
            results = cur.fetchall()
            
            relations = [(fact['source_entity_name'], fact['relationship_type'], fact['target_entity_name']) for fact in results]
            self.visualize(relations)
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                self.connection_pool.putconn(conn)
                
    #Planning, Algorithm 1
    def get_trajectory(self, valid_objects, task, observation, inventory, MAX_STEPS, MAX_QUERIES, sequence=None, reflection="", model='gpt-4-turbo'):
        """
        Generates a trajectory of a state-action-reward-state-action-reward (SARSA) sequences
        for a given task utilizing knowledge from a knowledge graph.

        Parameters:
        - valid_actions (list): A list of actions that can be performed in the environment.
                                Each action should be applicable to one or more objects
                                within the valid_objects list.
        - valid_objects (list): A list of objects that are present in the environment and
                                can be interacted with through the actions in valid_actions.
        - task (str): A string describing the task to be achieved. The task should be
                    interpretable based on the knowledge available in the knowledge graph,
                    guiding the trajectory generation.
        - observation (str): The initial observation

        Returns:
        - trajectory (list of tuples): A list where each element is a tuple representing a
                                    SARSA sequence: (state, action, reward, next_state, next_action).
                                    - state (dict): The current state of the environment, describing
                                                    the status of various objects.
                                    - action (str): The action taken in the current state.
                                    - reward (float): The reward received after taking the action.
                                    - next_state (dict): The state of the environment after the action
                                                            is taken.
                                    - next_action (str): The next action to be taken in the new state.
        """
        # Initialize the trajectory list and the initial state based on the observation
        state_0 = {
            'observation': observation,
            'inventory': inventory,
            'valid_receptacles': valid_objects,
        }
        
        env_sum = {
            'task': task,
            'reflection': reflection[-4:]
        }
        
        current_sars = {
                'state': state_0,
                'action': '<PREDICT>',
                'reward (env response)': '<PREDICT>',
                'next state': None,
                'done or termination': None
            }
        
        if not sequence:
            sequence = [
                current_sars
            ]
        else:
            sequence.append(current_sars)

        done = False
        step = 0
        plan_printout = ""
        
        while step < MAX_STEPS and not done:
            current_sars = sequence[-1]
            res = self.plan_next_action_reward(env_sum, sequence[-3:], model, max_query=MAX_QUERIES)
            current_sars['action'] = res['predicted_action']
            current_sars['reward (env response)'] = res['predicted_response']
            current_sars['next state'] = '<PREDICT>'
            current_sars['done or termination'] = '<PREDICT>'
            
            sequence[-1] = current_sars
            res = self.plan_next_state_termination(env_sum, sequence[-3:], model, max_query=MAX_QUERIES)
            current_sars['next state'] = res['state']
            current_sars['done or termination'] = res['task_completion']
            done = False
            
            sequence[-1] = current_sars
            next_sars = {
                'state': res['state'],
                'action': '<PREDICT>',
                'reward (env response)': '<PREDICT>',
                'next state': None,
                'done or termination': None
            }
            
            plan_printout += f"\n<Action> {sequence[-1]['action']} - <Response>{sequence[-1]['reward (env response)']}"     
            sequence.append(next_sars)
            
            step += 1
            
        return sequence, plan_printout, self.token_sent, self.token_received

    def plan_next_action_reward(self, env_sum, sequence, model, max_query=3):
        """
        Plans the next action and predicts its reward, incorporating reflections on past attempts.
        
        Parameters:
            env_sum (dict): Summary of the current environment state and task.
            sequence (list): The current sequence of SARSA tuples.
            reflection (list of str, optional): Reflections on past unsuccessful actions. Defaults to an empty list.
            max_query (int): The maximum number of KG queries allowed. Defaults to 5.
        
        Returns:
            dict: Predicted next action and its expected reward.
        """
        predicted = False
        num_query = 0
        kg_log = """"""
        while not predicted:
            action_gen_prompt = self.action_gen.format(env_sum, sequence, kg_log)
            res, token_count = retry(10, get_response, sm.ActionPrediction, model, action_gen_prompt, json=True, schema=sm.ActionPrediction, config=self.config)
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            if res['query']:
                num_query += 1
                if num_query >= max_query:
                    qa = f"""
QUERY: {res['query']}
RESPONSE: 'maximum number of query reached. Please pick an arbitrary action that is relevant to the task' 
                    """
                else: 
                    qa = f"""
QUERY: {res['query']}
RESPONSE: {self.query(res['query'])} 
                    """
                    
                # print(qa)
                kg_log += qa
            if res['predicted_action']:
                predicted = True
                # pprint(res)
        return res
        
    def plan_next_state_termination(self, env_sum, sequence, model, max_query=3):
        """
        Predicts the next state of the environment and whether the task is terminated.
        
        Parameters:
            env_sum (dict): Summary of the current environment state and task.
            sequence (list): The current sequence of SARSA tuples.
            max_query (int): The maximum number of KG queries allowed. Defaults to 5.
        
        Returns:
            dict: Predicted next state and task completion status.
        """
        print(model)
        kg_log = """"""
        predicted = False
        num_query = 0
        while not predicted:
            state_gen_prompt = self.state_gen.format(sequence, kg_log)
            res, token_count = retry(10, get_response, sm.StateWithQuery, model, state_gen_prompt, json=True, schema=sm.StateWithQuery, config=self.config)
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            if res['query']:
                num_query += 1
                if num_query >= max_query:
                    qa = f"""
QUERY: {res['query']}
RESPONSE: 'maximum number of query reached. Please pick an arbitrary action that is relevant to the task' 
                    """
                else: 
                    qa = f"""
QUERY: {res['query']}
RESPONSE: {self.query(res['query'])} 
                    """
                    
                # print(qa)
                kg_log += qa
            if res['state']:
                predicted = True
                # pprint(res)
        return res
    
    
    def memory_reset(self):
        """
        Clear all memories in KG

        Parameters:

        Returns:
            None
        """
        try:
            uuid = self.agent.this_uuid
            # Get a connection from the pool
            conn = self.connection_pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Disable foreign key checks (optional, depending on DB setup)
            cur.execute("SET session_replication_role = 'replica';")

            # Delete from fact_tuples table
            cur.execute("""
                DELETE FROM fact_tuples
                WHERE agent_uuid = %s;
            """, (uuid,))

            # Find entities associated with the deleted fact tuples
            cur.execute("""
                DELETE FROM entities
                WHERE id IN (
                    SELECT e.id
                    FROM entities e
                    LEFT JOIN fact_tuples ft ON e.id = ft.source_entity_id OR e.id = ft.target_entity_id
                    WHERE ft.agent_uuid = %s
                    GROUP BY e.id
                    HAVING COUNT(ft.id) = 0
                );
            """, (uuid,))

            # Clean up relationships if necessary
            cur.execute("""
                DELETE FROM relationships
                WHERE id NOT IN (
                    SELECT DISTINCT relationship_id FROM fact_tuples
                );
            """)

            # Re-enable foreign key checks
            cur.execute("SET session_replication_role = 'origin';")

            # Commit changes
            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error removing data by UUID: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)