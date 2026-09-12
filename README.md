# Agentic RAG — Enterprise Knowledge Assistant

A modular Agentic RAG system built with **FastAPI, CrewAI, LlamaIndex, PostgreSQL/pgvector, Docling, Ollama, Arize Phoenix, RAGAS, and OpenWebUI**.

This README documents the current working implementation, how to run it, the architecture, observability, evaluation plan, Azure deployment, and the remaining assignment work.

---

## 1. Architecture

```text
User / OpenWebUI
       |
       v
+-------------------+
|     FastAPI       |
|   /api/v1/chat    |
+---------+---------+
          |
          v
+-------------------+
|   ChatService     |
| memory + routing  |
+---------+---------+
          |
          v
+-------------------+
|   Router Agent    |
+----+---------+----+
     |         |
 general      RAG
               |
               v
        +--------------+
        | CrewAI Crew  |
        +------+-------+
               |
        +------+------+
        |             |
        v             v
 Research Agent   Answer Agent
        |
        v
 knowledge_base_search
        |
        v
+-------------------------+
|        RAG Pipeline     |
|                         |
| Query Embedding         |
|        |                |
|        v                |
| PGVector Top-10         |
|        |                |
|        v                |
| Reranking Top-3         |
|        |                |
|        v                |
| Citation Metadata       |
+------------+------------+
             |
             v
       Ollama qwen3:8b
             |
             v
       Grounded Answer
             |
             v
   PostgreSQL Conversation
          Memory

Observability:
FastAPI/Crew/RAG
       |
       v
OpenTelemetry
       |
       v
Phoenix :4317
       |
       v
Phoenix UI :6006
```

### Trace hierarchy

The application currently produces:

```text
chat
└── agentic_rag.crew
    └── rag.retrieval
```

The `chat` span records request/response information. The `crew` span records orchestration information. The `rag.retrieval` span records retrieval, reranking, scores, sources, and citation information.

---

## 2. Technology Stack

| Area | Technology |
|---|---|
| API | FastAPI |
| Agent orchestration | CrewAI |
| Document processing | Docling |
| RAG framework | LlamaIndex |
| Vector database | PostgreSQL + pgvector |
| Embeddings | Ollama `qwen3-embedding:0.6b` |
| LLM | Ollama `qwen3:8b` |
| Reranking | Custom lexical/vector hybrid reranker |
| Citations | Metadata-based citation manager |
| Memory | Current in-memory abstraction; PostgreSQL persistence planned |
| Observability | Arize Phoenix + OpenTelemetry |
| Evaluation | RAGAS |
| Frontend | OpenWebUI — planned |
| Deployment | Azure VM + Docker |

---

## 3. Repository Structure

```text
agentic-rag/
├── app/
│   ├── agents/
│   │   ├── crew.py
│   │   ├── crew_answer_agent.py
│   │   ├── rag_tool.py
│   │   ├── research_agent.py
│   │   ├── router_agent.py
│   │   └── tasks.py
│   │
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── evaluation/
│   │   ├── dataset.py
│   │   └── ragas_evaluator.py
│   │
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── memory/
│   │   └── conversation_memory.py
│   │
│   ├── observability/
│   │   ├── phoenix.py
│   │   └── prompts.py
│   │
│   ├── rag/
│   │   ├── embedding.py
│   │   ├── pgvector_store.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── citations.py
│   │
│   ├── services/
│   │   └── chat_service.py
│   │
│   ├── config.py
│   ├── main.py
│   └── __init__.py
│
├── documents/
│   ├── employee_handbook.pdf
│   ├── leave_and_attendance_policy.pdf
│   └── travel_and_expense_policy.pdf
│
├── data/
├── tests/
├── .env
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

---

# 4. Document Ingestion

Documents are processed with Docling.

```text
PDF
 |
 v
Docling DocumentConverter
 |
 v
Structured document
 |
 v
Markdown
 |
 v
Section-aware contextual chunking
 |
 v
Ollama embeddings
 |
 v
PostgreSQL + pgvector
```

OCR is disabled for the current PDFs because they contain selectable text.

Each chunk contains metadata such as:

```python
{
    "source": "leave_and_attendance_policy.pdf",
    "section": "Entitlement",
    "chunk_index": 0
}
```

The contextual chunk also contains document and section information before the original text.

### Current ingestion result

```text
employee_handbook.pdf              13 chunks
leave_and_attendance_policy.pdf    17 chunks
travel_and_expense_policy.pdf      14 chunks
-----------------------------------------
Total                               44 chunks
```

---

# 5. PostgreSQL + pgvector

PostgreSQL is deployed with:

```text
pgvector/pgvector:pg16
```

The vector extension is enabled:

```sql
CREATE EXTENSION vector;
```

The production RAG table is:

```text
rag_documents_1024
```

The dimension is **1024**, matching:

```text
qwen3-embedding:0.6b
```

Do not insert old 384-dimensional mock embeddings into the production 1024-dimensional table.

---

# 6. Retrieval

The current retrieval pipeline is:

```text
User query
    |
    v
Ollama embedding
    |
    v
1024-dimensional vector
    |
    v
pgvector similarity search
    |
    v
Top 10
```

Configuration:

```text
RETRIEVAL_TOP_K=10
RERANK_TOP_K=3
```

---

# 7. Reranking

The current reranker combines:

```text
70% lexical overlap
30% original vector similarity
```

Flow:

```text
PGVector
   |
   +-- candidate 1
   +-- candidate 2
   ...
   +-- candidate 10
            |
            v
       Reranker
            |
            v
       Top 3 results
```

The reranker is behind an abstraction so it can later be replaced with a cross-encoder.

---

# 8. Citations

Citations are generated from retrieval metadata rather than invented by the LLM.

A citation contains:

```text
citation_id
source
section
chunk_index
```

Example:

```text
[1] leave_and_attendance_policy.pdf — Entitlement
```

The answer agent is instructed to preserve citation markers such as:

```text
[1]
[2]
```

The application validates the citation IDs against actual retrieved citations and removes invalid citation numbers.

---

# 9. Agentic RAG

The system uses a deliberately small agent architecture.

## Router Agent

Decides:

```text
Question
   |
   +--> general
   |
   +--> rag
```

## Research Agent

Responsible for finding evidence.

Tool:

```text
knowledge_base_search
```

The tool performs:

```text
retrieval
   ↓
reranking
   ↓
citation creation
   ↓
evidence formatting
```

## Answer Agent

Transforms research evidence into a concise grounded answer.

Rules include:

- Use only research output.
- Do not invent facts.
- Preserve citations.
- Do not invent citation IDs.
- Do not perform another knowledge-base search.
- Do not include source filenames in the natural-language answer.

---

# 10. CrewAI

The current workflow is sequential:

```text
Research Task
      |
      v
Research Agent
      |
      v
knowledge_base_search
      |
      v
Research Evidence
      |
      v
Answer Task
      |
      v
Answer Agent
      |
      v
Final Answer
```

The RAG tool uses:

```python
result_as_answer = True
```

This prevents repeated research tool calls after evidence is returned.

CrewAI's independent telemetry is disabled because Phoenix/OpenTelemetry is used as the application observability layer:

```env
CREWAI_DISABLE_TELEMETRY=true
```

---

# 11. Ollama

Required models:

```text
qwen3:8b
qwen3-embedding:0.6b
```

Check:

```bash
ollama list
```

Check Ollama API:

```bash
curl http://localhost:11434/api/tags
```

Application endpoint:

```text
http://localhost:11434
```

---

# 12. FastAPI

Health endpoint:

```text
GET /health
```

Chat endpoint:

```text
POST /api/v1/chat
```

Example request:

```json
{
  "conversation_id": "demo-001",
  "message": "How many annual leave days do employees get?"
}
```

Example response:

```json
{
  "conversation_id": "demo-001",
  "answer": "Eligible full-time employees receive 20 days of annual leave per calendar year. [1]",
  "route": "rag",
  "reason": "Query requires information from the knowledge base.",
  "citations": [
    {
      "citation_id": 1,
      "source": "leave_and_attendance_policy.pdf",
      "section": "Entitlement",
      "chunk_index": 0
    }
  ]
}
```

Swagger:

```text
http://localhost:8000/docs
```

---

# 13. Docker Infrastructure

Current Docker services:

```text
PostgreSQL + pgvector
Phoenix
```

Start:

```bash
docker compose up -d postgres phoenix
```

Check:

```bash
docker ps
```

Expected containers:

```text
agentic-rag-postgres
agentic-rag-phoenix
```

Phoenix ports:

```text
6006 -> Web UI
4317 -> OTLP gRPC
```

---

# 14. Phoenix Observability

Phoenix is deployed with Docker.

Application configuration:

```env
PHOENIX_ENABLED=true
PHOENIX_ENDPOINT=http://localhost:4317
PHOENIX_PROJECT_NAME=agentic-rag
```

The Python application registers Phoenix before application components are imported.

The application uses explicit OpenTelemetry spans rather than relying on unavailable automatic OpenInference instrumentors.

### `chat` span

Attributes include:

```text
conversation.id
user.message
message.length
rag.route
rag.route_reason
rag.result
rag.citation_count
response.length
```

### `agentic_rag.crew` span

Attributes include:

```text
crew.query
crew.process
crew.agents
crew.output_length
crew.result
```

### `rag.retrieval` span

Attributes include:

```text
rag.query
rag.retrieval_top_k
rag.rerank_top_k
rag.retrieved_count
rag.reranked_count
rag.top_score
rag.citation_count
rag.result
```

Individual result attributes include:

```text
rag.result.0.source
rag.result.0.section
rag.result.0.score
rag.result.0.rerank_score
```

and equivalent attributes for results 1 and 2.

---

# 15. Phoenix UI

Open:

```text
http://<AZURE_PUBLIC_IP>:6006
```

Current Azure NSG requirement:

```text
TCP 6006 -> Allow
```

Keep OTLP `4317` internal whenever possible.

The application exports to:

```text
localhost:4317
```

---

# 16. Current Phoenix Trace

A successful application request produces:

```text
chat
└── agentic_rag.crew
    └── rag.retrieval
```

Example retrieval metadata:

```text
rag.retrieval_top_k = 10
rag.retrieved_count = 10
rag.rerank_top_k = 3
rag.reranked_count = 3
rag.top_score = 0.5576
rag.citation_count = 2
```

This provides visibility into retrieval quality, reranking, sources, and citations.

---

# 17. Conversation Memory

Current abstraction:

```text
conversation_id
    |
    +-- user message
    +-- assistant message
    +-- citations
    +-- timestamp
```

Current implementation is in-memory.

Planned implementation:

```text
ConversationMemory
       |
       v
PostgreSQL
```

This will allow conversations to survive application restarts and support multiple backend instances.

---

# 18. RAGAS Evaluation

The evaluation framework contains questions and ground truths for:

```text
How many annual leave days do employees get?

How many annual leave days can be carried forward?

What is the domestic hotel reimbursement limit?

How quickly must employees submit expenses?
```

Metrics:

```text
Faithfulness
Response Relevancy
Context Precision
Context Recall
```

The evaluator is already structured around RAGAS `SingleTurnSample`.

Remaining work is connecting the evaluator to an evaluation LLM and running the complete suite.

---

# 19. Prompt Management

The current project has:

```text
app/observability/prompts.py
```

with an application-level prompt version abstraction.

The next stage is to migrate this to Phoenix Prompt Management.

Target:

```text
Phoenix Prompt Registry
        |
        +-- rag_answer v1
        +-- rag_answer v2
        +-- production version
                    |
                    v
              Answer Agent
```

The goal is to support:

```text
create
retrieve
version
use
trace
```

---

# 20. OpenWebUI

OpenWebUI will provide the frontend.

Target:

```text
Browser
   |
   v
OpenWebUI
   |
   v
FastAPI
   |
   v
Agentic RAG
```

The backend remains responsible for:

```text
routing
agents
retrieval
reranking
memory
citations
LLM
observability
evaluation
```

---

# 21. Azure Deployment

Current VM:

```text
Name:
agentic-rag-vm

OS:
Ubuntu 24.04 LTS

Size:
Standard_D4as_v7

CPU:
4 vCPU

RAM:
16 GiB

Disk:
64 GiB Standard SSD

Region:
East US
```

The VM runs:

```text
FastAPI
Ollama
Docker
PostgreSQL
pgvector
Phoenix
```

Azure auto-shutdown is configured for cost control.

---

# 22. Environment Configuration

Recommended `.env`:

```env
APP_NAME=Agentic RAG
DEBUG=false

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=agentic_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIMENSIONS=1024

RETRIEVAL_TOP_K=10
RERANK_TOP_K=3

PHOENIX_ENABLED=true
PHOENIX_ENDPOINT=http://localhost:4317
PHOENIX_PROJECT_NAME=agentic-rag

CREWAI_DISABLE_TELEMETRY=true
```

Do not commit `.env` or private credentials.

---

# 23. Installation

Create the Conda environment:

```bash
conda create -n agentic-rag python=3.11 -y
conda activate agentic-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Important dependency note:

The current CrewAI environment has a known `tokenizers` compatibility warning related to the installed CrewAI version. Do not independently upgrade OpenTelemetry packages without checking CrewAI compatibility.

The working OpenTelemetry versions are:

```text
opentelemetry-api = 1.34.1
opentelemetry-sdk = 1.34.1
opentelemetry-semantic-conventions = 0.55b1
```

Phoenix packages currently used:

```text
arize-phoenix-otel = 0.17.1
arize-phoenix-client = 3.5.0
```

---

# 24. Start Everything

## Step 1 — PostgreSQL + Phoenix

```bash
docker compose up -d postgres phoenix
```

## Step 2 — Verify Ollama

```bash
ollama list
```

## Step 3 — Activate environment

```bash
conda activate agentic-rag
```

## Step 4 — Start FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Step 5 — Test health

```bash
curl http://localhost:8000/health
```

## Step 6 — Test RAG

```bash
curl -X POST http://localhost:8000/api/v1/chat   -H "Content-Type: application/json"   -d '{
    "conversation_id": "demo-001",
    "message": "How many annual leave days do employees get?"
  }'
```

---

# 25. Useful Test Questions

```text
How many annual leave days do employees get?

How many annual leave days can be carried forward?

What is the domestic hotel reimbursement limit?

How quickly must employees submit expenses?

What is the domestic travel class policy?

What are the company working hours?
```

---

# 26. Testing

Syntax:

```bash
python -m py_compile   app/services/chat_service.py   app/agents/rag_tool.py   app/agents/crew.py
```

Dependency consistency:

```bash
pip check
```

Tests:

```bash
pytest
```

API health:

```bash
curl http://localhost:8000/health
```

---

# 27. Git Commit Before Continuing Development

Check the working tree:

```bash
git status
```

Review changes:

```bash
git diff
```

Add files:

```bash
git add .
```

Commit the current milestone:

```bash
git commit -m "feat: add Phoenix observability to agentic RAG"
```

Push:

```bash
git push origin main
```

If the project uses another branch:

```bash
git branch --show-current
```

Then push that branch.

---

# 28. Recommended `.gitignore`

```gitignore
.env
.env.*
!.env.example

__pycache__/
*.py[cod]

.pytest_cache/
.mypy_cache/

.venv/
venv/

data/*
!data/.gitkeep

*.log

.DS_Store

.idea/
.vscode/

.ipynb_checkpoints/
```

Never commit:

```text
.env
Azure SSH private keys
API keys
production passwords
model files
database volumes
Phoenix database files
```

---

# 29. Assignment Status

| Requirement | Status |
|---|---|
| Docling document processing | Complete |
| PDF ingestion | Complete |
| Contextual chunking | Complete |
| LlamaIndex | Complete |
| PostgreSQL | Complete |
| pgvector | Complete |
| Ollama embeddings | Complete |
| Vector retrieval | Complete |
| Reranking | Complete |
| Citation metadata | Complete |
| Citation validation | Complete |
| Ollama LLM | Complete |
| CrewAI orchestration | Complete |
| Router Agent | Complete |
| Research Agent | Complete |
| Answer Agent | Complete |
| FastAPI API | Complete |
| Phoenix server | Complete |
| OpenTelemetry tracing | Complete |
| RAG observability | Complete |
| Conversation memory persistence | Pending |
| Phoenix Prompt Management | Next |
| Full RAGAS execution | Pending |
| OpenWebUI | Pending |
| Full Docker backend | Pending |
| Integration tests | Pending |
| Production hardening | Pending |

---

# 30. Development Roadmap

Continue from the current committed milestone in this order:

```text
1. Phoenix Prompt Management
          |
          v
2. PostgreSQL Conversation Memory
          |
          v
3. Complete RAGAS evaluation
          |
          v
4. OpenWebUI
          |
          v
5. Dockerize FastAPI
          |
          v
6. Full Docker Compose stack
          |
          v
7. Integration tests
          |
          v
8. Security/configuration hardening
          |
          v
9. Final assignment documentation
```

This order keeps the currently working RAG and Phoenix implementation stable while adding the remaining requirements incrementally.

---

# 31. End-to-End Example

For:

```text
How many annual leave days do employees get?
```

the runtime flow is:

```text
User
 |
 v
FastAPI
 |
 v
ChatService
 |
 v
Router Agent
 |
 +----> RAG
          |
          v
      Research Agent
          |
          v
  knowledge_base_search
          |
          v
   Query Embedding
          |
          v
     PGVector
          |
       Top 10
          |
          v
      Reranking
          |
        Top 3
          |
          v
   Citation Manager
          |
          v
    Research Result
          |
          v
      Answer Agent
          |
          v
      Ollama qwen3:8b
          |
          v
"Eligible full-time employees receive
20 days of annual leave per calendar year. [1]"
```

At the same time Phoenix records:

```text
chat
└── agentic_rag.crew
    └── rag.retrieval
```

---

# 32. Why This Architecture

The system is modular so each major component can be replaced independently.

For example:

```text
Ollama
  -> OpenAI / Azure OpenAI / another local model

SimpleReranker
  -> Cross-encoder

In-memory Memory
  -> PostgreSQL / Redis

PGVector
  -> another vector database

CrewAI
  -> another orchestration framework

Phoenix
  -> another observability backend
```

The abstractions around embeddings, reranking, memory, citations, and LLM providers reduce coupling between components.

---

# 33. Submission Notes

For assignment submission, demonstrate these parts:

1. PDF ingestion with Docling.
2. Contextual chunks and metadata.
3. PostgreSQL/pgvector retrieval.
4. Reranking.
5. CrewAI Router/Research/Answer workflow.
6. Ollama local LLM and embedding model.
7. Citation-grounded answers.
8. FastAPI API.
9. Phoenix trace showing:
   ```text
   chat
   └── agentic_rag.crew
       └── rag.retrieval
   ```
10. Phoenix retrieval attributes and source metadata.
11. RAGAS evaluation results once evaluation is completed.
12. OpenWebUI once frontend integration is completed.
13. Docker Compose for infrastructure.

---

## License

## Author

Md Manawar Iqbal