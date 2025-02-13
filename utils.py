import json
from openai import OpenAI, AzureOpenAI
import re
import os
import configparser
import psycopg2
from psycopg2 import pool, extras
from groq import Groq
import logging
from sklearn.metrics.pairwise import cosine_similarity

from octoai.client import Client
from pydantic import BaseModel, ValidationError
from pydub import AudioSegment
from time import sleep
from pydub.playback import play
import inspect
import random
import numpy as np

def get_connection_pool(config):
    """
    Creates and returns a database connection pool using psycopg2 based on provided configuration details.

    Parameters:
    - config (ConfigParser): An object containing database configuration details. Expected to have 'DB' section with 'DB_NAME', 'DB_USER', and 'DB_PASS' keys.

    Returns:
    - psycopg2.pool.SimpleConnectionPool: A connection pool object with a minimum of 1 connection and a maximum of 25 connections.

    Example usage:
    config = configparser.ConfigParser()
    config.read('config.ini')
    pool = get_connection_pool(config)
    """
    # Create a DB connection pool based on config details
    connection_pool = pool.SimpleConnectionPool(
        1,  # minconn
        400,  # maxconn
        dbname=config.get('DB', 'DB_NAME'),
        user=config.get('DB', 'DB_USER'),
        password=config.get('DB', 'DB_PASS')
    )

    return connection_pool

def count_connections(pool):
    try:
        return {
            'minconn': pool.minconn,
            'maxconn': pool.maxconn,
            'usedconn': pool._used,
            'freeconn': pool._pool
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_config(config_file):
    """
    Loads the config file from configuration file (ie: config.ini). See config.ini.example for details.
    :param config_file: path and filename for config.ini file
    :return: configparser.ConfigParser object or None if the file does not exist
    """
    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist.")
        return None

    config = configparser.ConfigParser()
    config.read(config_file)

    # Checking if the config file was empty or improperly formatted
    if not config.sections():
        print(f"Config file {config_file} is empty or improperly formatted.")
        return None

    return config

def setup_logger(config):
    """
    Setup logger for system.
    :param config: config object for use in setting up installation specific configuration variables.
    :return: logger object
    """
    # setup logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=config.get('DEFAULT', 'LOG_FILE'), format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    return logger

def estimate_token_usage(text):
    """
    Estimate the number of tokens in a given text using basic assumptions.
    
    Parameters:
    text (str): The text for which to estimate token usage.
    
    Returns:
    int: The estimated number of tokens.
    """
    # Remove any non-word characters to approximate token count
    words = re.findall(r'\w+', text)
    
    # Approximate number of tokens (considering punctuation, special characters as separate tokens)
    # Generally, average token length in GPT models is around 4 characters
    # Adding a factor to account for special tokens and punctuation
    estimated_tokens = sum(len(word) // 4 + 1 for word in words)
    
    return estimated_tokens

def openai_speak(config, text):
    api_key = config.get('DEFAULT', 'LLM_API_KEY')
    client = OpenAI(api_key=api_key)
    speech_file_path = "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )

    response.stream_to_file(speech_file_path)
    
    # load the file into pydub
    speech = AudioSegment.from_mp3(speech_file_path)
    
    play(speech)
    os.remove(speech_file_path)

def json_parser(config, input, max_attemps=5):
    system_prompt = f"""
You are JsonAI, your goal is to fix malformed json string.    
    
#### Instructions:
- **Structural Corrections**: Amend structural inaccuracies—balance brackets, ensure proper comma placement, and correct quotation usage.
- **Content Preservation**: Keep key-value pairs intact; do not alter data values or keys. Keep all keys intact! Do NOT add more keys or more values!
- **Eliminate Non-JSON Elements**: Remove characters or elements not recognized in JSON format.
- **Output**: Produce a clean, valid JSON object without commentary or extraneous text. I just need only the json object in the output.
- **Minimalism**: It is guaranteed that the orignial string only have very minimal error. So keep it minimal.

### Example Inputs and Outputs:

#### Input Malformed JSON:
```plaintext
"name": "John Doe", "age": 30, "city": "New York"
```

#### Output Corrected JSON:
```json
{{
  "name": "John Doe",
  "age": 30,
  "city": "New York"
}}
```

---

#### Input Malformed JSON:
```plaintext
{{,,,"name": "Jane Doe",, "age": 25,, "city": "Los Angeles",,,}}
```

#### Output Corrected JSON:
```json
{{
  "name": "Jane Doe",
  "age": 25,
  "city": "Los Angeles"
}}
```

---

#### Input Malformed JSON:
```plaintext
"favorites": "movies": ["Comedy", "Drama"], "books": ["Fiction", "Sci-fi"]
```

#### Output Corrected JSON:
```json
{{
  "favorites": {{
    "movies": ["Comedy", "Drama"],
    "books": ["Fiction", "Sci-fi"]
  }}
}}
```
    """
    content = {}
    pattern = r'`json\s*([\s\S]*?)\s*`'
    match = re.search(pattern, input)
    if match:
        input = match.group(1)
    llm_prompt = f"""    
    #### Input Malformed JSON:
    ```plaintext
    {input}
    ```

    #### Output Corrected JSON:
    """
    try:
        res = json.loads(input)
        res = {k.lower(): res[k] for k in res.keys()}
        return res
    except:
        attempt = 0
        while attempt < max_attemps: 
            try:
                # content = get_octoml_response(config, llm_prompt, model='meta-llama-3-70b-instruct', system_prompt=system_prompt)
                content = get_gpt_response(config, system_prompt + "\n" + llm_prompt, model='gpt-3.5-turbo', response_type='json_object')
                
                # azure
                # content = get_azure_response(config, system_prompt + "\n" + llm_prompt, model='gpt-4-turbo', response_type='json_object')
                match = re.search(pattern, content)
                if match:
                    res = json.loads(match.group(1))
                else:
                    res = json.loads(content)
                res = {k.lower(): res[k] for k in res.keys()}
                return res
            except Exception as e:
                print(f'try parsing json against: attempt #{attempt} \n {content}, receiving error: \n {e}')
                attempt += 1

def get_gpt_response(config,gpt_prompt,model, temperature=0.3, max_token=2000,response_type='text'):
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
    api_key = config.get('DEFAULT', 'LLM_API_KEY')
    client = OpenAI(api_key=api_key)


    messages = [{"role": "system", "content": gpt_prompt}]

    response_type_object = {"type": response_type}

    args = {
        "model": model,
        "temperature": temperature,  # description of temperature: http://bit.ly/3rmAJqu
        "max_tokens": max_token,
        "frequency_penalty": 0.0
    }
    
    if 'instruct' in model:
        args["prompt"] = gpt_prompt
        response = client.completions.create(**args)
        content = response.choices[0].text #The legacy way (as of NOv 8, 2023)
    else:
        if response_type == 'json_object':
            args["response_format"] = response_type_object
        args["messages"] = messages
        response = client.chat.completions.create(**args)
        content = response.choices[0].message.content

    #return json_objects
    return content

def get_octoml_response(config,llm_prompt,model='mixtral-8x7b-instruct', system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.", schema=None):
    octoml_api_token = config.get('DEFAULT', 'OCTOML_API_KEY')

    client = Client(token=octoml_api_token)
    args ={  
        "model":model,
        "max_tokens":1000,
        "presence_penalty":0,
        "temperature":0.2,
        "top_p":1,
        "messages":[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": llm_prompt},
        ]
    }

    if schema:
        args['response_format'] = {"type": "json_object",
                                   "schema": schema.model_json_schema()}

    completion = client.chat.completions.create(**args)

    # Handle the response from the API
    message_content = completion.choices[0].message.content
    
    if schema:
        message_content = json.loads(message_content)
    return message_content


def get_groq_response(config, llm_prompt, model='llama3-8b-8192', system_prompt=""):
    api_key = config.get('DEFAULT', 'GROQ_API_KEY')
    client = Groq(api_key=api_key)

    chat_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": llm_prompt},
        ]
    )
    return chat_response.choices[0].message.content

def get_nvidia_res(config, prompt, model="meta/llama3-70b"):
    api = config.get('DEFAULT', 'NVIDIA_API_KEY')
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = api
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=False
    )

    return completion.choices[0].message.content


def get_azure_response(config, gpt_prompt, model, temperature=0, max_token=2000, response_type='text'):
    """
    Use Azure OpenAI's API to generate a response to the given prompt.

    Parameters:
        config (ConfigParser): The configuration parser object that contains the API key and endpoint.
        gpt_prompt (str): The prompt to pass to the API.
        model (str): The model to use.
        temperature (float): Controls the randomness of the output. A higher value makes the output more random, while a lower value makes it more deterministic.
        max_token (int): Controls the maximum length of the generated response.
        response_type (str): The response format type, either 'text' or 'json_object'.

    Returns:
        str: The content of the response generated by the API.
    """
    
    if 'gpt-35-3' in model:        
        api_key = config.get('DEFAULT', 'AZURE_35_API_KEY')
        api_version = config.get('DEFAULT', 'AZURE_API_VER')
        azure_endpoint = config.get('DEFAULT', 'AZURE_35_ENDPOINT')
    else:
        api_key = config.get('DEFAULT', 'AZURE_API_KEY')
        api_version = config.get('DEFAULT', 'AZURE_API_VER')
        azure_endpoint = config.get('DEFAULT', 'AZURE_ENDPOINT')

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )

    messages = [{"role": "system", "content": gpt_prompt}]

    args = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_token,
        "frequency_penalty": 0.0
    }

    if response_type == 'json_object':
        args["response_format"] = {"type": response_type}

    while True:
        try:
            if 'instruct' in model:
                args["prompt"] = gpt_prompt
                response = client.completions.create(**args)
                content = response.choices[0].text  # The legacy way (as of Nov 8, 2023)
            else:
                args["messages"] = messages
                response = client.chat.completions.create(**args)
                content = response.choices[0].message.content

                # if 
            return content
        except Exception as e:
            error_details = json.loads(e.response.content)
            error_code = error_details.get('error', {}).get('code')
            if error_code == '429':
                print("Rate limit exceeded. Retrying in 2 seconds...")
                sleep(5)
            else:
                raise e



def get_response(model, prompt, json=False, schema=None, config=None, count_token=False):
    octo_models = {
        "qwen1.5-32b-chat",
        "meta-llama-3-8b-instruct",
        "meta-llama-3-70b-instruct",
        "mixtral-8x22b-instruct",
        "nous-hermes-2-mixtral-8x7b-dpo",
        "mixtral-8x7b-instruct",
        "mixtral-8x22b-finetuned",
        "hermes-2-pro-mistral-7b",
        "mistral-7b-instruct",
        "llamaguard-7b",
        "codellama-7b-instruct",
        "codellama-13b-instruct",
        "codellama-34b-instruct",
        "llama-2-13b-chat",
        "llama-2-70b-chat"
    }
    
    azure_models = {
        'gpt-35-3',
        # 'gpt-35-turbo-instruct',
        'gpt-40-1',
        'gpt-4-turbo-1'
    }
    
    gpt_models = {
        'gpt-4o',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-3.5',
        'gpt-3.5-turbo',
        'gpt-4o-mini'
    }
    
    groq_models = {
        'llama3-70b-8192',
        'llama3-8b-8192',
    }
    
    if model in gpt_models:
        res = get_gpt_response(config, prompt, model, temperature=0)
    elif model in azure_models:
        res = get_azure_response(config, prompt, model)
    elif model in octo_models:
        try: 
            res = get_octoml_response(config, prompt, model, schema=schema)
        except:
            res = get_octoml_response(config, prompt, model)
    elif model in groq_models: 
        res = get_groq_response(config, prompt, model)
    else:
        raise('unsupported model')
    if json:
        # res = json(res)
        res = json_parser(config, res)
    
    return res, {'sent': estimate_token_usage(prompt), 'received': estimate_token_usage(str(res))}

def get_embedding(config, text, model):
    """
    Fetches the embedding for a given text using a specified model.

    Args:
    config: A configuration object or dictionary containing API credentials and settings.
    text (str): The text string from which to generate an embedding.
    model (str): The model identifier used for generating embeddings.

    Returns:
    list: A list representing the text embedding if successful, None otherwise.
    
    Raises:
    Exception: If the API call fails or returns an error.
    """
    # Replace newlines in the text with spaces to ensure compatibility with the embedding model
    text = text.replace("\n", " ")

    # Try to fetch the embedding using the provided model and text
    try:
        api_key = config.get('DEFAULT', 'LLM_API_KEY')
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=[text], model=model)

        # Check if the response is successful and contains the expected data
        if response.data:
            return response.data[0].embedding
        else:
            print("Failed to retrieve embeddings. The response was unsuccessful or contained no data.")
            return None
    except Exception as e:
        print(f"An error occurred while fetching embeddings: {str(e)}")
        raise

def get_cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def retry(max_attempts, func, val_func, *args, **kwargs):
    """
    Retry a function up to max_attempts times until it returns a valid output
    as specified by a validation function or a Pydantic model schema.

    Parameters:
    - max_attempts (int): The maximum number of attempts to make.
    - func (callable): The function to call, which should return a dictionary
                    that can be validated by the validation function or Pydantic model.
    - val_func (callable or BaseModel): The validation function or Pydantic model class used for validating the function output.
    - args, kwargs: Arguments and keyword arguments to pass to the function.

    Returns:
    - dict: The valid output from the function as per the validation function or Pydantic model.

    Raises:
    - ValueError: If the function does not produce a valid output after the maximum number of attempts.
    - Exception: Propagates any exception raised by the function that is not related to validation.
    """
    attempt = 0
    token_count = {'sent': 0, 'received': 0}
    while attempt < max_attempts:
        result, new_token_count = func(*args, **kwargs)
        token_count['sent'] += new_token_count['sent']
        token_count['received'] += new_token_count['received']
        try:
            if inspect.isclass(val_func) and issubclass(val_func, BaseModel):
                # Validate the result using the provided Pydantic model
                validated_result = val_func(**result)
                return validated_result.dict(), token_count  # Return the validated data as a dictionary
            else:
                # Validate the result using the provided validation function
                if val_func(result):
                    return result, token_count
        except ValidationError as ve:
            print(result)
            print(f"Validation failed on attempt {attempt + 1}: {ve}")
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            
        print(f"Validation failed on attempt {attempt + 1}: {result}")
        attempt += 1

    return "", token_count
    # raise ValueError(f"Function did not produce a valid output after {max_attempts} attempts")

#code by Lin et al
def load_variation(env, set, cutoff=False):
    variations = []
    if (set == "train"):
        variations = list(env.getVariationsTrain())
        # if task_num == 26: 
        #     variations = variations[:int(len(variations)/10)]
        # elif task_num == 29: 
        #     variations = variations[:int(len(variations)/2)]
    elif (set == "test"):
        variations = list(env.getVariationsTest())
        if cutoff:
            test_len = min(50, len(variations))
            random.seed(1)
            random.shuffle(variations)
            variations = variations[:test_len]
    elif (set == "dev"):
        variations = list(env.getVariationsDev()) 
        variations = variations[:3]
    elif (set == "test_mini_2"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[3:10] 
    elif (set == "test_mini"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[:3] 
    elif (set == "test_mini_mini"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[:1] 
    else:
        raise("ERROR: Unknown set to evaluate on (" + str(set) + ")")
        # logger.info("ERROR: Unknown set to evaluate on (" + str(set) + ")")
        exit(1)
 
    # logger.info(variations)
    return variations

def findValidActionNew(predictions, env, look, recent_actions, sbert_model, logger, k=5):
    focus_on_count = {
    "0": 1, "1": 1, "2": 1, "3": 1, "4": 2, "5": 1, "6":1, "7":1,
    "8": 1, "9": 1, "10": 1, "11": 1, "12": 4, "13": 4, "14":1, "15":1,
    "16": 1, "17": 1, "18": 2, "19": 1, "20": 3, "21": 3, "22":1, "23":1,   
    "24": 1, "25": 1, "26": 2, "27": 1, "28": 1, "29": 2
    
    }

    rooms = ["hallway", "greenhouse", "green house", "kitchen", "bathroom", "outside", "workshop", "art studio", "foundry", "bedroom", "living room"]

    valid_open_door = ["open door to " + i for i in rooms] 
    invalid_focus = ["focus on "+x for x in ["agent", "air"]+rooms]
    validActions = set(env.getValidActionObjectCombinations())
    validActions.update(valid_open_door)
    validActions.difference_update(invalid_focus)

    inventory = env.inventory().lower()
    
    validActions.difference_update(recent_actions[-3:]) 

    for va in list(validActions):
        if "door" in va and "open" not in va:
            validActions.remove(va)
            continue
        if va.startswith("focus on"): 
            pattern = re.compile(r"\b(?:focus|on|in|to)\b", re.IGNORECASE)
            used_objs = pattern.sub("", va).split(" ")
            valid = True
            for obj in used_objs:
                if obj not in look + " " + inventory:
                    valid = False
            if not valid:
                validActions.remove(va)
    

    # 1) if acton in top k is valid, choose it
    found_valid_in_top = False
    action = None
    for pred in predictions[:k]:
        pred = pred.replace("green house", "greenhouse") 
        if pred.strip() in validActions:
            found_valid_in_top = True
            action = pred.strip()
            break
    if found_valid_in_top:
        return action 
    else:
        logger.info(f"No valid action found in top k={k} predictions.")
        validActions = list(validActions)
        validActions.sort(key=lambda x: len(x))
 

    # 2) else, find most similar action

    if sbert_model:    
        pred_vectors = sbert_model.encode(predictions[:5], batch_size=5, show_progress_bar=False)
        valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)


        # Calculate cosine similarity between each vector in pred_vectors and all vectors in valid_action_vectors
        similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)

        # Take the sum of cosine similarities for each vector in valid_action_vectors
        sum_similarities = similarity_matrix.sum(axis=0)

        # Find the indices of the k vectors with the highest sum of cosine similarities
        N = 5 # Change this to the number of top vectors you want to retrieve
        top_indices = np.argpartition(sum_similarities, -N)[-N:]

        # Print the indices of the top vectors
        # print(f"The indices of the top {k} vectors in valid_action_vectors are: {top_indices}")
        logger.info("The most similar valid actions to the predictions:")
        for ti in top_indices:
            logger.info("\t\t - "+validActions[ti])
        action = validActions[top_indices[0]]
    else:
        # jaccard
        topValue = 0.0
        topAction = predictions[0]
        # embPred = sbert_model.encode(pred, convert_to_tensor=True)
        tokensPred = predictions[0].split(" ")
        uniqueTokensPred = set(tokensPred)

        for validAction in validActions: 
            tokensAction = validAction.split(" ")
            uniqueTokensAction = set(tokensAction)

            intersection = uniqueTokensPred.intersection(uniqueTokensAction)
            if (len(intersection) > topValue):
                topAction = validAction
                topValue = len(intersection)

        logger.info("TOP VALID ACTION: " + topAction)
        # Sanitize top action
        topAction = re.sub(r'[^A-Za-z0-9 ]+', '', topAction)
        action = topAction
    return action 
