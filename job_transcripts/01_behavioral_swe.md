---
interview_id: 01_behavioral_swe
participant: Anthony Chan (candidate, self-recorded)
role: Behavioral interview prep — SWE intern
interview_type: behavioral
date: 2026-04-10
---

INTERVIEWER: Tell me about a time you worked with a difficult team member.

CANDIDATE: Yeah, so this was last semester, I was on a group project for our distributed systems class. There were four of us. One person — I'll call him D — basically wasn't doing his part. He'd miss meetings, his code wouldn't compile when he pushed it, and when we'd try to talk to him about it he'd get defensive. So um, what I did was, first I tried to just talk to him one-on-one, like grab coffee, see if something was going on in his life. Turned out he was taking six classes and working twenty hours a week, which is just too much. So I helped him reduce his scope on the project to one piece we knew he could handle, and the rest of us picked up his other parts. It worked out. We got an A.

INTERVIEWER: That's good. What did you learn from that?

CANDIDATE: I learned that the easy thing is to be frustrated with someone, and the harder thing is to figure out what's actually going on. Most people aren't slacking because they don't care, they're slacking because something else is going on.

INTERVIEWER: Okay. Tell me about a project you're proud of.

CANDIDATE: So I built — I'm building actually — this thing called Verbatim. It's a tool that lets you have a conversation with customer research interviews. You ingest a folder of transcripts and you can ask things like what are the top reasons users churned, and it gives you answers with verbatim quotes and citations.

INTERVIEWER: How does it work technically?

CANDIDATE: So under the hood it's a RAG system but I went a step beyond — instead of just retrieving once and dumping into a prompt, I built an agent loop. Claude has four tools — semantic search, full transcript loading, exact phrase search, and a corpus index. The agent decides which to call. And then I wrapped the whole thing with an eval harness with planted ground truth so I could measure whether each iteration of the pipeline actually got better.

INTERVIEWER: Why did you generate the corpus synthetically?

CANDIDATE: Yeah, that was the choice I'm probably most proud of. Real customer research transcripts aren't publicly available, and even if I scraped some, I wouldn't know the ground truth for what's in them. By generating the corpus I controlled what every interview said, which meant my evals could measure exact precision and recall, not just "did the LLM judge like the answer." Trustworthy evals beat real data here.

INTERVIEWER: What did you find when you ran the evals?

CANDIDATE: Um, so the agent pipeline beat naive RAG by a decent margin, especially on cross-interview synthesis questions. Naive RAG would only see its top-K chunks and miss interviews that didn't surface; the agent could go back and search again. But — and this is the honest part — I don't remember the exact numbers off the top of my head, I'd have to pull them up.

INTERVIEWER: That's fine. What would you do differently if you started over?

CANDIDATE: I'd version the prompts more carefully. I changed the agent's system prompt like four times during development and I didn't track which prompt version produced which eval score. So if a score got worse I couldn't always tell if it was the retrieval change or the prompt change. Real prompt versioning is something I want to learn.

INTERVIEWER: Good answer. Last question — why this internship?

CANDIDATE: Because the job post matched what I'm interested in. They listed semantic search over interviews, MCP, and evals as example projects, and I'd basically built a small version of all three. I want to do the work I'm reading about in their post, not adjacent work.
