# Verbatim — Setup Guide

A step-by-step walkthrough for getting Verbatim running on your machine.

## Prerequisites

You need:
- **Python 3.10+** (check with `python3 --version`)
- A terminal you're comfortable with
- **~10 minutes** for setup, **~$5** for Anthropic API credit (hard ceiling, no auto-charge), free for Voyage

## Step 1 — Get into the project folder

```bash
cd ~/path/to/verbatim
ls
```

You should see folders like `src/`, `transcripts/`, `evals/`, `docs/`, `job_transcripts/`
and files like `README.md`, `pyproject.toml`.

## Step 2 — Create a Python virtual environment

This keeps the project's dependencies isolated from your system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked when your terminal prompt starts with `(.venv)`.
**Every time you open a new terminal in this project**, you need to run
`source .venv/bin/activate` again.

## Step 3 — Install dependencies

```bash
pip install -e .
```

This reads `pyproject.toml` and installs everything (anthropic, voyageai,
chromadb, streamlit, mcp, python-dotenv). Takes 1–2 minutes.

## Step 4 — Verify the stub eval works (no keys needed)

This proves the project is installed correctly before you spend time on
API keys.

```bash
python -m src.test_eval
python -m src.test_chunker
python -m src.eval --pipeline stub
```

You should see:
- 12 + 8 = 20 tests pass
- Stub eval shows `MEAN 97.9% 95.1% 96.0%`

If anything fails here, fix it before going further.

## Step 5 — Get an Anthropic API key

1. Go to https://console.anthropic.com
2. Sign in (or sign up — free to create an account)
3. Click **Settings → Billing → Add to credit balance**. Add **$5**.
   - This is a *prepaid balance*, NOT a subscription.
   - Anthropic does NOT auto-charge you. The $5 is a hard ceiling.
   - Don't toggle auto-reload. It's off by default.
4. Go to **Settings → API Keys → Create Key**
5. Copy the key — it starts with `sk-ant-`. They only show it once;
   save it somewhere safe immediately.

## Step 6 — Get a Voyage AI key (for embeddings)

1. Go to https://www.voyageai.com
2. Sign up
3. API Keys → Create. Copy the key.

Voyage's free tier is 200M tokens. This project uses well under 100k. You
will not pay anything for Voyage.

## Step 7 — Create your .env file

```bash
cp .env.example .env
```

Then open `.env` in your editor and paste in the real values:

```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
VOYAGE_API_KEY=pa-your-real-key-here
```

**Important:** no quotes around the keys, no spaces around `=`.

Verify `.env` is gitignored (it is, but worth knowing):

```bash
grep "^.env$" .gitignore
```

## Step 8 — Build the vector index

This embeds all 10 transcripts and stores them locally.

```bash
make ingest
# or: python -m src.ingest
```

You should see:
```
Ingesting from: /path/to/transcripts
Collection:     interviews
Indexed 198 chunks in 3.2s.
Vector store persisted at: /path/to/.chroma
```

If it errors with "VOYAGE_API_KEY not set," your `.env` isn't being read
or the key isn't there. Re-check Step 7.

## Step 9 — Run the real evals

This is the moment of truth. First the baseline:

```bash
make eval-naive
```

Then the agent:

```bash
make eval-agent
```

Compare them:

```bash
make eval-compare
```

You'll see a side-by-side table. The agent should beat naive RAG on F1
overall, especially on multi-interview questions. Save the output —
that table is the thing you screenshot for your README and demo video.

## Step 10 — Launch the chat UI

```bash
make app
```

Streamlit opens a browser tab at `http://localhost:8501`. Try the example
questions in the sidebar. Switch between `agent_v1` and `naive_rag` to
show the difference live — this is great B-roll for a demo video.

## Step 11 — (optional) Set up the MCP server in Claude Desktop

If you have Claude Desktop installed:

1. Find the config file:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Edit it to include Verbatim:
   ```json
   {
     "mcpServers": {
       "verbatim": {
         "command": "/absolute/path/to/verbatim/.venv/bin/python",
         "args": ["-m", "src.mcp_server"],
         "cwd": "/absolute/path/to/verbatim"
       }
     }
   }
   ```
   Replace `/absolute/path/to/verbatim` with the real absolute path
   (run `pwd` in your terminal to get it).

3. Restart Claude Desktop. You should see a tools/plug icon indicating
   Verbatim is connected.

4. Try asking Claude Desktop: *"Use Verbatim to tell me the top reasons
   trial users are churning."*

## Step 12 — (optional) Job-interview extension

The same architecture runs over the job-interview corpus:

```bash
make ingest-jobs
python -m src.eval --pipeline agent_v1 \
    --ground-truth evals/job_ground_truth.json
```

This proves the architecture is domain-agnostic.

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
You're not in the project root, or you're not running with `python -m`.
`cd` into the `verbatim/` folder and use `python -m src.eval`, not
`python src/eval.py`.

**`ModuleNotFoundError: No module named 'anthropic'`**
Your venv isn't activated. Run `source .venv/bin/activate`.

**`ANTHROPIC_API_KEY not set` or `VOYAGE_API_KEY not set`**
Your `.env` file is missing, malformed, or in the wrong folder. It must
be in the project root and named exactly `.env`.

**`AuthenticationError: 401`**
Your API key is wrong or has no credits. Check console.anthropic.com → Billing.

**Eval scores are weirdly low**
Two common causes: (1) you didn't run `make ingest`, so the vector
store is empty; (2) you ingested a different corpus on top. Run
`make clean && make ingest` to rebuild.
