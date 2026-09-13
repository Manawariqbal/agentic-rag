# Agentic RAG --- Enterprise Knowledge Assistant

A modular Agentic RAG system built with **FastAPI, CrewAI, LlamaIndex,
PostgreSQL/pgvector, Docling, Ollama, Arize Phoenix, RAGAS, and
OpenWebUI**.

The current implementation supports document ingestion, contextual
chunking, vector retrieval, reranking, deterministic evidence gating,
citation-aware answers, PostgreSQL-backed conversation memory, CrewAI
orchestration, Phoenix observability/prompt management, and an
OpenAI-compatible API consumed by OpenWebUI.

------------------------------------------------------------------------

## 1. Architecture

``` text
OpenWebUI
    |
    v
FastAPI /v1/chat/completions
    |
    v
ChatService
    |
    +--------------------+
    |                    |
    v                    v
Router Agent         Conversation Memory
    |
    +----> general
    |
    +----> RAG
              |
              v
        Retrieval Query
        (current question)
              |
              v
        LlamaIndex Retriever
              |
              v
        PostgreSQL + pgvector
              |
              v
          Top 10
              |
              v
           Reranker
              |
              v
           Top 3
              |
              v
        Evidence Gate
          /       \
       reject      pass
         |          |
         v          v
     Abstain    CrewAI Crew
                    |
             +------+------+
             |             |
             v             v
       Research Agent  Answer Agent
             |
             v
       Retrieved Evidence
             |
             v
        Grounded Answer
             |
             v
        Citation Manager
             |
             v
       PostgreSQL Memory

Observability:
FastAPI / CrewAI / RAG
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

### Important grounding boundary

Retrieval, reranking, and evidence gating happen **before** CrewAI.

The deterministic reranked evidence is explicitly passed into the CrewAI
research task. This prevents the Research Agent from independently
deciding to skip the retrieval tool and hallucinating enterprise policy
information.

------------------------------------------------------------------------

## 2. Technology Stack

  Area                  Technology
  --------------------- ---------------------------------------
  API                   FastAPI
  Agent orchestration   CrewAI
  Document processing   Docling
  RAG framework         LlamaIndex
  Vector database       PostgreSQL + pgvector
  Embeddings            Ollama `qwen3-embedding:0.6b`
  LLM                   Ollama `qwen3:8b`
  Reranking             Custom lexical/vector hybrid reranker
  Citations             Metadata-based citation manager
  Evidence control      Deterministic Evidence Gate
  Memory                PostgreSQL-backed conversation memory
  Observability         Arize Phoenix + OpenTelemetry
  Evaluation            RAGAS
  Frontend              OpenWebUI
  Deployment            Azure VM + Docker Compose

------------------------------------------------------------------------

## 3. Repository Structure

``` text
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
│   │   ├── citations.py
│   │   ├── evidence_gate.py
│   │   └── rag_prompt.py
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
├── .env.docker
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 4. Document Ingestion

Documents are processed with Docling.

``` text
PDF
 |
 v
Docling DocumentConverter
 |
 v
Structured document
 |
 v
Section-aware contextual chunking
 |
 v
Contextual chunks
 |
 v
Ollama embeddings
 |
 v
PostgreSQL + pgvector
```

Current documents:

``` text
employee_handbook.pdf
leave_and_attendance_policy.pdf
travel_and_expense_policy.pdf
```

Current ingestion result:

``` text
employee_handbook.pdf              13 chunks
leave_and_attendance_policy.pdf    17 chunks
travel_and_expense_policy.pdf      14 chunks
-----------------------------------------
Total                               44 chunks
```

The contextual chunker uses approximately:

``` text
chunk size: 800
overlap:    100
```

Chunks retain metadata such as:

``` python
{
    "source": "leave_and_attendance_policy.pdf",
    "section": "Entitlement",
    "chunk_index": 0
}
```

The contextual prefix includes document and section information to
improve retrieval quality.

------------------------------------------------------------------------

## 5. PostgreSQL + pgvector

PostgreSQL is deployed using:

``` text
pgvector/pgvector:pg16
```

The vector extension is enabled in PostgreSQL.

The current production table is:

``` text
rag_documents_1024
```

The embedding dimension is:

``` text
1024
```

matching:

``` text
qwen3-embedding:0.6b
```

Do not insert vectors generated with a different embedding dimension
into the production table.

------------------------------------------------------------------------

## 6. Retrieval Pipeline

The current retrieval pipeline is:

``` text
Current user question
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
Top 10 candidates
        |
        v
Reranker
        |
        v
Top 3 results
        |
        v
Evidence Gate
```

Configuration:

``` env
RETRIEVAL_TOP_K=10
RERANK_TOP_K=3
RAG_RELEVANCE_THRESHOLD=0.48
```

### Conversation-aware retrieval

Conversation memory is still used for routing and downstream
conversational context.

However, the **vector retrieval query uses the current user message
directly**.

This avoids contaminating semantic retrieval with previous turns.

Example:

``` text
Previous:
How many annual leave days do employees get?

Current:
How many unused days can be carried forward?
```

The vector search receives:

``` text
How many unused days can be carried forward?
```

rather than the entire previous conversation.

------------------------------------------------------------------------

## 7. Reranking

The current reranker combines:

``` text
70% lexical overlap
30% original vector similarity
```

Flow:

``` text
PGVector Top 10
       |
       v
   Reranker
       |
       v
    Top 3
```

The reranker is abstracted so it can later be replaced by a
cross-encoder or another reranking model.

------------------------------------------------------------------------

## 8. Evidence Gate

The Evidence Gate is a deterministic anti-hallucination boundary.

``` text
Reranked Results
       |
       v
Top rerank score
       |
       +---- score < 0.48 ----> Abstain
       |
       +---- score >= 0.48 ---> CrewAI
```

When evidence is insufficient, the backend returns:

``` text
I don't have enough information in the available company policies to answer this question.
```

CrewAI is not called in that case.

This prevents unsupported enterprise-policy answers from reaching the
generation layer.

------------------------------------------------------------------------

## 9. Citations

Citations are generated from retrieval metadata rather than invented by
the LLM.

A citation contains:

``` text
citation_id
source
section
chunk_index
```

Example:

``` text
[1] leave_and_attendance_policy.pdf — Entitlement
```

The answer agent is instructed to preserve citation markers such as:

``` text
[1]
[2]
```

The application validates citation IDs against the actual retrieved
citation set and removes invalid citation numbers.

------------------------------------------------------------------------

## 10. Agentic RAG

The system uses a deliberately small agent architecture.

### Router Agent

Routes requests to:

``` text
Question
   |
   +--> general
   |
   +--> rag
```

The router is conversation-aware for follow-up questions.

Examples of policy-related follow-up terms include:

``` text
those
that
it
they
them
these
this
previous
earlier
carry forward
carried forward
unused days
```

### Research Agent

The Research Agent is responsible for interpreting retrieved evidence.

The deterministic retrieval pipeline runs before CrewAI. The retrieved
evidence is explicitly supplied to the research task.

The agent is instructed:

-   Use only supplied retrieved evidence.
-   Do not use its own knowledge.
-   Do not invent facts.
-   Do not invent document names.
-   Do not invent section names.
-   Do not invent page numbers.
-   Do not invent citations.
-   State when evidence is insufficient.

### Answer Agent

The Answer Agent transforms research evidence into a concise grounded
response.

Rules include:

-   Use only the research result.
-   Do not invent facts.
-   Preserve valid citations.
-   Do not invent citation IDs.
-   Do not perform another knowledge-base search.
-   Do not include source filenames in the natural-language answer.
-   Do not include a separate Sources section in the agent-generated
    answer.

The API layer can append the validated citation/source presentation
returned by the citation manager.

------------------------------------------------------------------------

## 11. CrewAI

The current CrewAI workflow is sequential:

``` text
Deterministic Retrieval
        |
        v
Evidence Gate
        |
        v
Research Task
        |
        v
Research Agent
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

The deterministic retrieved/reranked evidence is passed into the
Research Task.

This is intentional: the application does not depend on the LLM deciding
whether it should call `knowledge_base_search`.

The RAG tool remains available as part of the agent architecture for
future agentic retrieval workflows.

CrewAI telemetry is disabled because Phoenix/OpenTelemetry is used as
the application's observability layer:

``` env
CREWAI_DISABLE_TELEMETRY=true
```

------------------------------------------------------------------------

## 12. Ollama

Required models:

``` text
qwen3:8b
qwen3-embedding:0.6b
```

Check models:

``` bash
ollama list
```

Check Ollama:

``` bash
curl http://localhost:11434/api/tags
```

Application configuration:

``` env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIMENSIONS=1024
```

For Docker, the backend uses:

``` env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

------------------------------------------------------------------------

## 13. FastAPI

Health endpoint:

``` text
GET /health
```

Application chat endpoint:

``` text
POST /api/v1/chat
```

OpenAI-compatible model endpoint:

``` text
GET /v1/models
```

OpenAI-compatible chat endpoint:

``` text
POST /v1/chat/completions
```

Example request:

``` json
{
  "conversation_id": "demo-001",
  "message": "How many annual leave days do employees get?"
}
```

Example response:

``` json
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

``` text
http://localhost:8000/docs
```

------------------------------------------------------------------------

## 14. OpenWebUI

OpenWebUI is deployed through Docker Compose:

``` text
Browser
   |
   v
OpenWebUI :3000
   |
   v
Backend :8000
   |
   v
Agentic RAG
```

Current frontend port:

``` text
3000 -> OpenWebUI container port 8080
```

The backend exposes an OpenAI-compatible API so OpenWebUI can use the
Agentic RAG service as a model endpoint.

Current Azure frontend:

``` text
http://<AZURE_PUBLIC_IP>:3000
```

------------------------------------------------------------------------

## 15. Conversation Memory

Conversation memory is PostgreSQL-backed.

The abstraction stores:

``` text
conversation_id
role
message
citations
timestamp
```

The current chat flow:

``` text
User message
    |
    v
Load recent messages
    |
    v
Conversation-aware routing
    |
    v
Store user message
    |
    v
RAG / general processing
    |
    v
Store assistant answer + citations
```

Recent conversation history is used for follow-up routing, while the
current question is kept as the direct vector retrieval query.

------------------------------------------------------------------------

## 16. Phoenix Observability

Phoenix is deployed using Docker.

Ports:

``` text
6006 -> Phoenix Web UI
4317 -> OTLP gRPC
```

The application uses explicit OpenTelemetry spans.

Current trace hierarchy:

``` text
chat
└── agentic_rag.crew
    └── rag.retrieval
```

### `chat` span

Tracks information such as:

``` text
conversation.id
user.message
message.length
rag.route
rag.route_reason
rag.retrieval_query
rag.retrieved_count
rag.reranked_count
rag.top_rerank_score
rag.evidence_sufficient
rag.evidence_threshold
rag.evidence_gate_reason
rag.citation_count
response.length
```

### `agentic_rag.crew` span

Tracks:

``` text
crew.query
crew.process
crew.agents
crew.has_retrieved_evidence
crew.output_length
crew.result
```

Prompt metadata includes:

``` text
prompt.name
prompt.version_id
prompt.model
prompt.provider
prompt.template_format
```

### `rag.retrieval` span

Tracks:

``` text
rag.query
rag.retrieval_top_k
rag.rerank_top_k
rag.retrieved_count
rag.reranked_count
rag.top_score
rag.citation_count
rag.result
```

Individual retrieval result metadata includes source, section, score,
and rerank score.

------------------------------------------------------------------------

## 17. Phoenix Prompt Management

The current answer-generation task retrieves the RAG prompt through:

``` text
app/rag/rag_prompt.py
```

The prompt is retrieved from the Phoenix-managed prompt abstraction and
its system instructions are injected into the Answer Task.

The target prompt lifecycle is:

``` text
Phoenix Prompt Registry
        |
        +-- rag_answer versions
        |
        v
Retrieve production prompt
        |
        v
Answer Agent
        |
        v
Phoenix trace
```

This supports the project requirement of prompt retrieval/version
visibility while keeping prompt logic outside the core chat service.

------------------------------------------------------------------------

## 18. RAGAS Evaluation

The evaluation framework contains questions and ground truths for:

``` text
How many annual leave days do employees get?

How many annual leave days can be carried forward?

What is the domestic hotel reimbursement limit?

How quickly must employees submit expenses?
```

Target metrics:

``` text
Faithfulness
Response Relevancy
Context Precision
Context Recall
```

The evaluator is structured around RAGAS `SingleTurnSample`.

Remaining evaluation work includes connecting the evaluator to the
evaluation LLM and running the complete suite.

------------------------------------------------------------------------

## 19. Docker Compose

Current services:

``` text
agentic-rag-postgres
agentic-rag-phoenix
agentic-rag-backend
agentic-rag-openwebui
```

Start the full stack:

``` bash
docker compose up -d
```

Check:

``` bash
docker compose ps
```

Expected services:

``` text
postgres       Healthy
phoenix        Running
backend        Up
openwebui      Healthy
```

PostgreSQL:

``` text
5432 -> 5432
```

Backend:

``` text
8000 -> 8000
```

OpenWebUI:

``` text
3000 -> 8080
```

Phoenix:

``` text
6006 -> 6006
4317 -> 4317
```

------------------------------------------------------------------------

## 20. Local Development

Create the environment:

``` bash
conda create -n agentic-rag python=3.11 -y
conda activate agentic-rag
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start infrastructure:

``` bash
docker compose up -d postgres phoenix
```

Run the backend:

``` bash
uvicorn app.main:app --reload
```

------------------------------------------------------------------------

## 21. Docker Development

Build backend:

``` bash
docker compose build backend
```

Start backend:

``` bash
docker compose up -d backend
```

View logs:

``` bash
docker logs -f agentic-rag-backend
```

Check service health:

``` bash
docker compose ps
```

Compile-check a Python file inside the backend container:

``` bash
docker exec agentic-rag-backend python -m py_compile app/services/chat_service.py
```

------------------------------------------------------------------------

## 22. Environment Configuration

Host `.env` example:

``` env
APP_NAME=Agentic RAG
DEBUG=true

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

CREWAI_DISABLE_TELEMETRY=true
RAG_RELEVANCE_THRESHOLD=0.48
```

Docker `.env.docker` uses:

``` env
POSTGRES_HOST=postgres
OLLAMA_BASE_URL=http://host.docker.internal:11434
PHOENIX_ENDPOINT=http://phoenix:4317
```

Never commit:

``` text
.env
.env.docker
```

if they contain real credentials or secrets.

Use `.env.example` for shareable configuration.

------------------------------------------------------------------------

## 23. Azure Deployment

Current VM:

``` text
Name:
agentic-rag-vm

OS:
Ubuntu 24.04 LTS x64 Gen2

Size:
Standard_D4as_v7

CPU:
4 vCPU

RAM:
16 GiB

Disk:
128 GiB

Region:
East US

Public IP:
Configured separately
```

The VM runs:

``` text
Docker
Ollama
FastAPI
PostgreSQL
pgvector
Phoenix
OpenWebUI
```

Azure auto-shutdown is configured for cost control.

The public-facing application ports currently used are:

``` text
3000 -> OpenWebUI
8000 -> FastAPI
6006 -> Phoenix UI
```

Keep database and OTLP ports restricted to trusted networks where
possible.

------------------------------------------------------------------------

## 24. Verified RAG Behaviors

The current implementation has been tested against the following
important cases.

### Known policy question

Question:

``` text
How many annual leave days do employees get?
```

Result:

``` text
Eligible full-time employees receive 20 days of annual leave per calendar year. [1]
```

Citation:

``` text
leave_and_attendance_policy.pdf — Entitlement
```

### Follow-up question

After asking about annual leave:

``` text
How many unused days can be carried forward?
```

Result:

``` text
Up to 5 unused annual-leave days may be carried into the next leave year. [1]
```

Citation:

``` text
leave_and_attendance_policy.pdf — Carry Forward
```

### Unknown policy question

Question:

``` text
What is the employee stock option vesting policy?
```

Result:

``` text
I don't have enough information in the available company policies to answer this question.
```

The system correctly abstains instead of inventing a policy.

------------------------------------------------------------------------

## 25. Key Engineering Decisions

### Deterministic retrieval before agent generation

The system does not rely on an LLM to decide whether enterprise evidence
exists.

### Current-question retrieval

Conversation history is preserved for context, but the current user
question is used directly for vector retrieval.

### Evidence gate

Low-confidence retrieval results are rejected before generation.

### Evidence passed into CrewAI

CrewAI receives the actual deterministic retrieved evidence instead of
independently hallucinating enterprise facts.

### Metadata-based citations

Citations originate from document metadata and retrieved chunks.

### PostgreSQL-backed memory

Conversation state survives backend restarts and can support future
multi-instance deployments.

### Local model hosting

Ollama keeps LLM and embedding inference under the application's
infrastructure rather than requiring an external model API.

------------------------------------------------------------------------

## 26. Current Project Status

### Completed

-   [x] Docling document processing
-   [x] Contextual chunking
-   [x] Ollama embeddings
-   [x] PostgreSQL + pgvector
-   [x] LlamaIndex retrieval
-   [x] Top-K retrieval
-   [x] Reranking
-   [x] Citation metadata
-   [x] Conversation memory
-   [x] Conversation-aware routing
-   [x] Deterministic evidence gate
-   [x] CrewAI Research Agent
-   [x] CrewAI Answer Agent
-   [x] Deterministic evidence passed into CrewAI
-   [x] Phoenix tracing
-   [x] Phoenix prompt retrieval
-   [x] RAGAS evaluation structure
-   [x] FastAPI
-   [x] OpenAI-compatible API
-   [x] OpenWebUI Docker deployment
-   [x] Azure deployment
-   [x] Known-answer grounding test
-   [x] Follow-up retrieval test
-   [x] Unknown-question abstention test

### Remaining / Next

-   [ ] Complete RAGAS evaluation run
-   [ ] Improve evaluation dataset
-   [ ] Add automated integration tests
-   [ ] Add cross-encoder reranking
-   [ ] Expand agentic routing/research behavior
-   [ ] Add richer Phoenix dashboards and prompt versioning workflows
-   [ ] Harden production security
-   [ ] Add authentication/authorization
-   [ ] Add CI/CD
-   [ ] Add production deployment documentation
-   [ ] Add performance/load testing

------------------------------------------------------------------------

## 27. Demo Test Cases

### Leave

``` text
How many annual leave days do employees get?
```

``` text
How many unused days can be carried forward?
```

``` text
How many sick leave days are available?
```

### Travel

``` text
What is the domestic hotel reimbursement limit?
```

``` text
Are taxes included in the hotel limit?
```

``` text
How quickly must employees submit expenses?
```

``` text
What class of flight is allowed for domestic travel?
```

### Employee handbook

``` text
What are the normal working hours?
```

``` text
What does the remote work policy say?
```

### Hallucination test

``` text
What is the employee stock option vesting policy?
```

``` text
What is the employee referral bonus?
```

``` text
Does the company provide car loans?
```

The system should abstain when the requested information is not
supported by the indexed company documents.

------------------------------------------------------------------------

## 28. Project Goal

The project demonstrates an end-to-end production-oriented Agentic RAG
architecture combining:

``` text
Document Intelligence
        +
Contextual Retrieval
        +
Vector Search
        +
Reranking
        +
Evidence Gating
        +
Agentic Orchestration
        +
Conversation Memory
        +
Citation Grounding
        +
Prompt Management
        +
Observability
        +
Evaluation
        +
OpenAI-Compatible API
        +
OpenWebUI
        +
Docker
        +
Azure
```

The emphasis is on **grounded answers, observable agent execution,
explicit evidence boundaries, reproducible local model hosting, and
production-oriented architecture**.
