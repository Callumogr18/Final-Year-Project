# LLM Evaluation Framework

A framework for evaluating large language models using traditional NLP metrics (BLEU, ROUGE), LLM-as-Judge scoring, and a hybrid evaluation approach. The project compares traditional similarity-based metrics against judge-based qualitative evaluation across task types: QA, Summarisation, and Reasoning.

## Project Structure

```
source/
├── DB/                    # Database layer (connection, prompt & response managers, seeding)
├── LLM/                   # LLM clients, response generation, judge logic
├── metrics/
│   ├── traditional/       # BLEU / ROUGE scorers
│   └── hybrid/            # Hybrid scoring (traditional + judge combined)
├── tests/                 # pytest test suite
├── main.py                # Interactive CLI evaluation runner
├── api.py                 # FastAPI async evaluation API
├── dashboard.py           # Streamlit results dashboard
├── compose.yaml           # Docker Compose for local PostgreSQL
└── requirements.txt
```

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for local database)
- Access to an Azure AI Foundry / Azure OpenAI resource

---

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\Activate.ps1     # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root. All keys below are required.

```dotenv
# --- Database ---
# Local Docker instance (used when running locally)
DATABASE_URL="postgresql://<user>:<password>@localhost:5433/<db_name>"

# --- Azure OpenAI inference models ---
# Shared API key for all inference endpoints
API_KEY="<your-azure-api-key>"
API_V="2025-01-01-preview"

# GPT-4.1-mini
GPT_ENDPOINT=AZURE ENDPOINT
GPT_MODEL="gpt-4.1-mini"

# Grok
GROK_ENDPOINT=AZURE ENDPOINT
GROK_MODEL="grok-4-1-fast-non-reasoning"

# LLaMA
LLAMA_ENDPOINT=AZURE ENDPOINT
LLAMA_MODEL="Llama-3.3-70B-Instruct"

# Phi
PHI_ENDPOINT=AZURE ENDPOINT
PHI_MODEL="Phi-4"

# Text embeddings (used in hybrid scoring)
EMBEDDING_ENDPOINT=AZURE ENDPOINT
EMBEDDING_MODEL="text-embedding-3-small"

# --- LLM-as-Judge (separate Azure deployment) ---
JUDGE_ENDPOINT=AZURE ENDPOINT
JUDGE_KEY="<your-judge-api-key>"
JUDGE_MODEL="<deployed-judge-model-name>"
JUDGE_API_V="2024-05-01-preview"
```

> **Note:** `API_KEY` is shared across GPT, Grok, LLaMA, and Phi endpoints. The judge uses a **separate** key (`JUDGE_KEY`) pointing to a different Azure deployment.

---

## Database Setup

### 1. Start PostgreSQL with Docker Compose

The `compose.yaml` exposes PostgreSQL on port **5433** (to avoid conflicts with any local PostgreSQL on 5432).

Create a `.env` (or export these variables) for the Docker container itself:

```dotenv
UNAME=fyp_user
PASSWD=root
DB=postgres
```

Then start the container:

```bash
docker compose up -d
```

Verify it is running:

```bash
docker ps
```

### 2. Create the schema

Connect to the database and run the following DDL to create the required tables:

```sql
CREATE TABLE public.prompts (
    id          SERIAL PRIMARY KEY,
    task_type   TEXT NOT NULL,
    question    TEXT,
    ground_truths TEXT[],
    answer      TEXT,
    contexts    TEXT[],
    article     TEXT,
    highlights  TEXT
);

CREATE TABLE public.generations (
    response_id     UUID PRIMARY KEY,
    prompt_id       INTEGER REFERENCES public.prompts(id),
    model_name      TEXT,
    llm_response    TEXT,
    latency         FLOAT,
    tokens_generated INTEGER,
    tokens_prompt   INTEGER,
    total_tokens    INTEGER,
    created_at      TIMESTAMP
);

CREATE TABLE public.metrics (
    id          SERIAL PRIMARY KEY,
    response_id UUID REFERENCES public.generations(response_id),
    prompt_id   INTEGER REFERENCES public.prompts(id),
    task_type   TEXT,
    batch_id    UUID,
    bleu        FLOAT,
    rouge1      FLOAT,
    rouge2      FLOAT,
    rougeL      FLOAT
);

CREATE TABLE public.judge_metrics (
    id              SERIAL PRIMARY KEY,
    response_id     UUID REFERENCES public.generations(response_id),
    prompt_id       INTEGER REFERENCES public.prompts(id),
    task_type       TEXT,
    batch_id        UUID,
    hallucination   FLOAT,
    fluency         FLOAT,
    consistency     FLOAT,
    reasoning       FLOAT,
    coherence       FLOAT,
    factual_accuracy FLOAT
);
```

You can connect with any PostgreSQL client, e.g.:

```bash
psql "postgresql://fyp_user:root@localhost:5433/postgres"
```

### 3. Seed the database

The seed script pulls evaluation datasets from HuggingFace and inserts them into the `prompts` table. It loads samples from:

| Dataset       | Task Type     | Sample Size |
| ------------- | ------------- | ----------- |
| SQuAD         | QA            | 100         |
| HotpotQA      | QA            | 100         |
| TruthfulQA    | QA            | 100         |
| AdversarialQA | QA            | 100         |
| CNN/DailyMail | Summarisation | 25          |

```bash
python -m DB.seed_data
```

> **Warning:** The seed script truncates all four tables (`prompts`, `generations`, `metrics`, `judge_metrics`) before inserting. Do not run it against a database with results you want to keep.

---

## Running the Application

### Interactive CLI

Runs the full evaluation pipeline interactively — prompts you to select by prompt ID, task type, or a list of IDs.

```bash
python main.py
```

### FastAPI Service

Starts the async REST API on `http://localhost:8000` and visit the /docs page for Swagger UI.

```bash
uvicorn api:app --reload
```

Key endpoints:

| Method | Path                    | Description                       |
| ------ | ----------------------- | --------------------------------- |
| `GET`  | `/health`               | Liveness check                    |
| `GET`  | `/prompts?task_type=QA` | List prompts by task type         |
| `POST` | `/evaluate`             | Start a background evaluation job |

### Streamlit Dashboard

Visualises stored results from the database.

```bash
streamlit run dashboard.py
# or via the CLI entrypoint:
python main.py dashboard
```

---

## Running Tests

```bash
pytest
```

The test suite covers traditional metric scoring, hybrid scoring, LLM cost/latency behaviour, pydantic judge models, and prompt manager logic.

---

## Azure API Configuration Notes

- All four inference models (GPT, Grok, LLaMA, Phi) use the **OpenAI-compatible endpoint** (`/openai/v1/`) and share one `API_KEY`.
- The judge model uses **LangChain's `AzureChatOpenAI`** and requires a **separate endpoint and key** (`JUDGE_ENDPOINT`, `JUDGE_KEY`, `JUDGE_API_V`).

# Live Deployments

There is also live deployments for interaction with the evaluation engine and FastAPI Swagger UI

FastAPI - https://my-fastapi-production-0e1a.up.railway.app/
Streamlit - https://final-year-project-production-a8e0.up.railway.app/
