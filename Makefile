.PHONY: help install test ingest ingest-jobs eval-stub eval-naive eval-agent eval-compare app mcp clean

help:
	@echo "Verbatim — available commands"
	@echo ""
	@echo "  make install         install dependencies into the active venv"
	@echo "  make test            run unit tests (chunker + eval math)"
	@echo ""
	@echo "  make ingest          embed the customer-research corpus (needs VOYAGE_API_KEY)"
	@echo "  make ingest-jobs     embed the job-interview corpus into a separate collection"
	@echo ""
	@echo "  make eval-stub       run evals against the stub (no API keys needed)"
	@echo "  make eval-naive      run evals against naive RAG (needs both API keys)"
	@echo "  make eval-agent      run evals against the agent (needs both API keys)"
	@echo "  make eval-compare    A/B compare naive_rag vs agent_v1"
	@echo ""
	@echo "  make app             launch the Streamlit demo UI"
	@echo "  make mcp             run the MCP server (for Claude Desktop integration)"
	@echo ""
	@echo "  make clean           remove .chroma/ and __pycache__/"

install:
	pip install -e .

test:
	python -m src.test_eval
	python -m src.test_chunker

ingest:
	python -m src.ingest

ingest-jobs:
	python -m src.ingest --dir job_transcripts --collection jobs

eval-stub:
	python -m src.eval --pipeline stub --save

eval-naive:
	python -m src.eval --pipeline naive_rag --save

eval-agent:
	python -m src.eval --pipeline agent_v1 --save

eval-compare:
	python -m src.eval --compare naive_rag agent_v1 --save

app:
	streamlit run src/app.py

mcp:
	python -m src.mcp_server

clean:
	rm -rf .chroma __pycache__ src/__pycache__ evals/runs/*.json
