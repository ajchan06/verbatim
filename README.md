# Verbatim

> Talk to your customer research interviews. Agentic search with citations, eval-driven from day one.

Ingests a folder of customer-research interview transcripts and lets you ask
questions across all of them — getting answers backed by verbatim quotes
with citations.

```
> What's the top reason trial users are churning?

Top reasons trial users churn are too many onboarding setup steps
(Sarah, Marcus, Tom), pricing felt steep for small teams (Marcus,
Yuki), and a poor mobile app experience (Priya).

Citations:
• [02_marcus] "I sat down on a Tuesday night to get everyone set up
   and an hour later I was still in settings..."
• [03_priya] "Half the things I can do on web I can't do on mobile..."
• [01_sarah] "By the time I got to the third or fourth screen I kind
   of just wanted to send a link to someone..."
```

---

## The five-minute pitch

**The problem.** Companies run hundreds of hours of customer-research
interviews. Those transcripts live in Drive folders. Nobody re-reads them.
The insights die there.

**The product.** Verbatim is an agentic-search tool that turns that
graveyard into a conversation. Ask it anything about the corpus —
"who churned because of pricing," "compare what power users love vs
trial users hated," "find me three quotes about the mobile app" —
and it returns sourced answers with verbatim citations.

**The shape.** Four layers:
1. **Chunker** — speaker-turn chunking with surrounding-turn context windows
2. **Vector store** — Voyage embeddings in a local Chroma index
3. **Agent** — Claude Sonnet with four tools (`search_interviews`,
   `get_full_transcript`, `find_quotes`, `list_interviews`)
4. **Eval harness** — twelve ground-truth questions, scored on precision,
   recall, F1, and LLM-as-judge faithfulness

It's also exposed as an **MCP server**, so Claude Desktop can connect
directly and query the corpus.

---

## Why this design

A few choices worth flagging up front, since they're the things a
reviewer should ask about:

### 1. Eval-first development

Most AI demos are built first and maybe evaluated later. This one was
evaluated first. Before any retrieval pipeline existed, the eval
harness was built and verified against a deliberately-broken stub. The
stub had planted recall misses and precision misses; the harness caught
them in exactly the right way. *Then* I built the real pipelines, with
the harness already in place to measure each iteration honestly.

This is the unintuitive but important move. Without evals, "the new
version feels better" is the best you can do. With evals, every change
to chunking, embedding, prompts, or tools is a measurable delta.

### 2. Synthetic corpus

Real customer-research transcripts at scale aren't publicly available.
The few academic datasets (Princeton gig-worker corpus, MediaSum) are
either tiny or wrong-domain. So the corpus is synthetic.

This is *better*, not worse: because I generated it, I know the ground
truth for every claim. Each interview has planted facts (e.g. "F5: the
Salesforce integration is broken, mentioned by interviews 06/08/09").
The eval set tests against those exact expected answers. Trustworthy
evals beat real data with mushy ground truth.

### 3. Agent over naive RAG

I built naive RAG first as a baseline — retrieve top-K chunks, stuff
into a prompt, generate. The agent isn't just a fancier version; it's
a deliberate response to naive RAG's known failure modes:

- Naive RAG is bounded by K. If a multi-interview question has five
  right answers but the top-K chunks come from three interviews, it
  misses the other two.
- The agent can do follow-up searches when a first pass is incomplete.
- It has a `list_interviews` tool that gives it a corpus map, so it can
  pick targets instead of blind-searching.

Whether the agent actually wins is an empirical question, which the eval
harness answers. (See `evals/runs/` for the numbers.)

### 4. Speaker-turn chunking, not character chunking

Most RAG tutorials chunk by fixed character count. That splits insights
mid-thought and dilutes retrieval. Interview transcripts have a natural
unit — the speaker turn. Each turn becomes one chunk, and each chunk
includes the previous and next turn as context. This matters because
interviewer questions often contain keywords ("Why did you cancel?")
that the participant doesn't repeat in the answer.

---

## Setup

```bash
# 1. Create and activate a venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install deps
pip install -e .

# 3. Configure API keys
cp .env.example .env
# Then edit .env and paste in your ANTHROPIC_API_KEY and VOYAGE_API_KEY.

# 4. Build the vector index (embeds all 10 interviews; takes ~5 seconds)
make ingest
```

You need two API keys:
- **Anthropic** (https://console.anthropic.com) — Claude is the LLM that answers questions and judges faithfulness. Add $5 in credits at Settings → Billing. Hard ceiling, no auto-charge. This project uses ~$3 total.
- **Voyage AI** (https://www.voyageai.com) — embeddings. Free tier covers this easily.

---

## Running it

### As a chat UI

```bash
make app   # streamlit run src/app.py
```

Open the URL Streamlit prints and ask away. The sidebar has example questions
and a pipeline selector so you can compare naive RAG vs the agent live.

### As an MCP server (in Claude Desktop)

```bash
make mcp
```

Then add this to your `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "verbatim": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/verbatim"
    }
  }
}
```

Restart Claude Desktop and the four Verbatim tools will appear. Ask
"what are users saying about pricing" and Claude will call the tools.

### As a CLI (for evals)

```bash
make eval-stub        # no API keys needed — proves the harness works
make eval-naive       # naive RAG baseline
make eval-agent       # the agent
make eval-compare     # A/B them side-by-side
```

Results are persisted to `evals/runs/<pipeline>_<timestamp>.json` so you
can diff runs and find regressions.

---

## Repo layout

```
verbatim/
├── docs/dataset_design.md       # why synthetic, what facts are planted where
├── evals/
│   ├── ground_truth.json        # 12 customer-research eval questions
│   ├── job_ground_truth.json    # 6 questions for the job-interview corpus
│   └── runs/                    # eval results, persisted for diffing
├── src/
│   ├── types.py                 # Pipeline Protocol, Citation, PipelineAnswer
│   ├── chunker.py               # speaker-turn chunking
│   ├── vector_store.py          # Voyage + Chroma wrapper
│   ├── ingest.py                # one-shot CLI to build the index
│   ├── stub_pipeline.py         # rule-based stub; verifies the eval harness
│   ├── naive_rag.py             # retrieve-once baseline
│   ├── agent.py                 # tool-using LLM with 4 tools
│   ├── mcp_server.py            # exposes the agent's tools via MCP
│   ├── judge.py                 # LLM-as-judge faithfulness scoring (Haiku)
│   ├── app.py                   # Streamlit chat UI
│   ├── eval.py                  # the eval harness
│   ├── test_eval.py             # unit tests for scoring math
│   └── test_chunker.py          # unit tests for chunking
├── transcripts/                 # 10 customer-research interviews
├── job_transcripts/             # 6 mock job interviews (the extension)
├── pyproject.toml
├── Makefile
├── .env.example
└── README.md
```

---

## The job-interview extension

The system is domain-agnostic. To prove it, the same backbone runs over a
second corpus — six mock job-interview transcripts I recorded for my own
interview prep. Same chunker, same vector store, same agent, same eval
harness. Different ground truth (`evals/job_ground_truth.json`), different
collection (`jobs`).

```bash
make ingest-jobs                                                    # build a separate index
python -m src.eval --pipeline agent_v1 \
  --ground-truth evals/job_ground_truth.json                        # run job-corpus evals
```

This was the small extension that demonstrated the architecture's
flexibility. The same code that helps a product manager grok customer
interviews helps a candidate grok their own mock interviews and spot
weak patterns. ("In which interviews did I self-identify a weakness?
What was my stock answer for 'tell me about a failure' and what was
the better one?")

---

## What's next (the honest "things I'd do with more time" section)

A weekend project, not a product. Real next steps:

1. **Prompt versioning.** I changed prompts during development and lost
   track of which version produced which score. The version of every
   prompt should be tagged so an eval run knows the prompt revision.
2. **Reranking.** Pull more candidates with vector search, then rerank
   with a cross-encoder before stuffing into context. Likely bumps
   precision a meaningful amount.
3. **Fan-out for synthesis questions.** "Top three themes" should
   run multiple sub-queries explicitly rather than relying on a single
   retrieval. The agent does some of this already; making it
   first-class would help.
4. **Per-user access controls.** Customer interviews are sensitive;
   any production version needs row-level access.
5. **Incremental ingestion.** Currently a rebuild is full. New
   interviews should update the index without re-embedding everything.
6. **Real-corpus validation.** Synthetic data validated the
   architecture; running this on a partner's real corpus is the
   honest next test.

---

## Acknowledgments

Built as a portfolio project. Inspired by Great Question's product
direction (customer research as searchable, queryable infrastructure).
The synthetic corpus, fictional product "Linkup," and all interview
participants are fabricated.
