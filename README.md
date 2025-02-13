# DAVIS: Guiding Scientific Agent Through Knowledge Graph-Powered Inner Monologue

## Introduction

Designing a generalist scientific agent capable of performing tasks in laboratory settings to assist researchers has become a key goal in recent Artificial Intelligence (AI) research. Unlike everyday tasks, scientific tasks are inherently more delicate and complex, requiring agents to possess a higher level of reasoning ability, structured and temporal understanding of their environment, and a strong emphasis on safety. Existing approaches often fail to address these multifaceted requirements. To tackle these challenges, we present DAVIS\footnote{The name is inspired by the first author's institution}. Unlike traditional retrieval-augmented generation (RAG) approaches, DAVIS incorporates structured and temporal memory, which enables model-based planning. Additionally, DAVIS implements an agentic, multi-turn retrieval system, similar to human's inner monologue, allowing for a greater degree of reasoning over past experiences. Through internal planning before each step, DAVIS significantly reduces the likelihood of taking unsafe actions compared to baseline models. DAVIS demonstrates competitive performance on the ScienceWorld benchmark, showcasing its ability to effectively leverage memory for reasoning and decision-making in dynamic scientific environments.

## How to Reproduce
### Steps
1. **Clone the Repository**
   ```
   git clone https:/<REDACTED>/ReasonPlanner
   cd ReasonPlanner
   ```

2. **Install Dependencies (Python 3.11.0)**
   ```
   pip install -r requirements.txt
   ```

3. **Create PostgreSQL Tables**

   Run the following query to create the required tables:

   ```bash
   psql -U your_username -d your_database -f kg_graph/kgraph.psql
   ```

4. **Configure the Project**
   - Fill in `config/config.ini.example` with your API keys, PostgreSQL username, and password.
   - Rename `config/config.ini.example` to `config.ini`

5. **Populate and Construct WorldModel**

   Run the training script:

   ```bash
   python ReasonAgentTraining.py
   ```

   This process might take a while.

6. **Run the Experiments (COSTLY)**

   Ensure you have at least $50 of OpenAI credits available. Short tasks typically take up to 5 minutes, while long tasks can take up to an hour. **Full 90 variations over 30 tasks will take up to $2000 of credits.**

   ```bash
   python ExperimentRunner.py
   ```
