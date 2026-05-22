---
interview_id: 05_project_deep_dive
participant: Anthony Chan (candidate, self-recorded)
role: Deep dive on Verbatim project — AI engineering focus
interview_type: technical
date: 2026-04-18
---

INTERVIEWER: Walk me through Verbatim, starting from the problem.

CANDIDATE: Sure. The problem is that companies do customer research interviews — sometimes hundreds of hours of them — and that data dies in a Drive folder. Nobody re-reads it, and the insights are locked up. Verbatim is a tool that lets you have a conversation with that folder. You ask things like what's the top reason users churned, and it gives you an answer with verbatim quotes and citations back to the original interviews.

INTERVIEWER: How is it different from just stuffing all the transcripts into Claude's context window?

CANDIDATE: A few reasons. One, real interview corpuses are way bigger than even Claude's two-hundred-thousand-token context. Two, even when they fit, you pay for every token and you get worse retrieval — needles get lost in haystacks. Three, you can't audit which interviews influenced an answer if they're all just mashed in. Citations matter because researchers can't quote a paraphrase in a stakeholder report; they need verbatim text with a source.

INTERVIEWER: Walk me through the architecture.

CANDIDATE: Four layers. First, chunking — I chunk transcripts by speaker turn instead of by character count, because a turn is the natural unit of meaning. Each chunk includes the previous and next turn as context, which helps retrieval match on conversational context, not just literal content. Second, embeddings and vector search — Voyage embeddings into a local Chroma store. Third, the agent — Claude with four tools: semantic search, full transcript loading, exact phrase search, and a corpus index. The agent decides what to call. Fourth, eval harness — twelve ground-truth questions with planted facts, measuring precision, recall, F1, and faithfulness via LLM-as-judge.

INTERVIEWER: Why an agent instead of just naive RAG?

CANDIDATE: I built naive RAG first as a baseline. The problem with naive RAG is that it's bounded by K. If you ask "which interviewees mentioned X" and there are five right answers but your top-ten chunks all come from three of those interviews, you miss the other two. The agent can do a search, see that it found three interviews, and then either search again with a different query or pull the corpus index to check if it's missing anyone. Naive RAG can't do that. When I ran the evals the agent beat naive RAG on multi-interview retrieval by a meaningful margin.

INTERVIEWER: How did you decide which tools to give the agent?

CANDIDATE: I started by looking at the eval questions and asking, for each one, what would a good researcher need? For "what's the top churn reason" you need to search broadly. For "Chloe's team size" you'd benefit from just loading Chloe's transcript. For "who said the exact phrase forty-eight bucks" you need keyword search. For "are there team admins in the corpus" you need a directory. So the four tools fell out of the question types. I deliberately didn't add tools the questions didn't justify.

INTERVIEWER: What about evals — what's the design?

CANDIDATE: The corpus is synthetic, which I'll explain why in a second. Because it's synthetic I know the ground truth for every claim. So for each eval question I have the exact set of interview_ids that should be returned. Precision is fraction of returned interviews that are correct; recall is fraction of expected interviews that were returned; F1 is the harmonic mean. For synthesis questions I also use LLM-as-judge with Haiku to check if the answer is actually supported by the cited excerpts.

INTERVIEWER: Why synthetic data?

CANDIDATE: Two reasons. One, real customer research transcripts at any scale aren't publicly available, and the ones that exist are wrong-domain. Two, synthetic data lets me plant known facts. I can say "fact F5 is that the Salesforce integration is broken, mentioned by interviews 06, 08, 09" and then I know the exact expected answer to "who said Salesforce was broken." Trustworthy evals beat real data with mushy ground truth.

INTERVIEWER: What was the hardest part?

CANDIDATE: Building the eval harness before any real pipeline existed. The temptation is always to build the cool thing first. But I forced myself to build the harness against a deliberately-broken stub pipeline first, so I could verify the scoring math worked. When I planted recall misses and precision misses in the stub, the harness caught them in exactly the right way. That gave me confidence that any later failure I saw was a real pipeline issue, not an eval bug.

INTERVIEWER: What would you do next if you had more time?

CANDIDATE: Three things. One, prompt versioning — I changed prompts during development and lost track of which version produced which score. Two, retrieval reranking — pull more candidates with vector search then rerank with a cross-encoder, would likely bump precision. Three, fan-out for synthesis questions — when the question is "what are the top three themes," explicitly run multiple sub-queries and synthesize rather than relying on a single retrieval.

INTERVIEWER: Last one — what's the production version of this look like?

CANDIDATE: At scale you'd want incremental ingestion so new interviews update the index without a rebuild. You'd want per-user access controls because customer interviews are sensitive. You'd want better observability — every tool call logged, every prompt version tagged, latency histograms. You'd probably want a proper rerank stage and you'd want to do user feedback collection so the system can learn over time. But the core shape — chunk, embed, agent with tools, evals — that's what production looks like. The demo is just the small version.
