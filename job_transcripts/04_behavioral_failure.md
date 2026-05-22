---
interview_id: 04_behavioral_failure
participant: Anthony Chan (candidate, self-recorded)
role: Behavioral — tell me about a failure
interview_type: behavioral
date: 2026-04-16
---

INTERVIEWER: Tell me about a time you failed.

CANDIDATE: Okay. So freshman year I took a really hard math class, math 55 equivalent, and I bombed the first midterm. Like I got a 40. I'd been used to being the smartest in the room in high school and suddenly I wasn't. I considered dropping. But I stuck with it, started going to office hours, formed a study group, and by the end of the semester I got a B. So I failed, but I came back.

INTERVIEWER: Hmm. Is there a more recent example, maybe something work or project related?

CANDIDATE: Yeah, that's fair, that one was kind of stock. Let me think. Okay — last summer I was building a side project, a small social app for my friend group, and I shipped it without doing any testing. Just pushed to production and texted people the link. Within an hour someone had figured out you could read other users' messages by changing a number in the URL. It was an IDOR vulnerability, classic stuff. I had to take it down immediately. I felt terrible.

INTERVIEWER: What did you learn from that?

CANDIDATE: A few things. One, "move fast and break things" is fine when you're the only one breaking, not when other people's data is involved. Two, security isn't a separate concern you bolt on at the end, it's something you have to design for. Three, even toy projects have real-world consequences if real people use them. I rebuilt it with proper auth checks before relaunching and I had a friend try to break it before I told anyone the link.

INTERVIEWER: Good. Why was that a better example than the math class story?

CANDIDATE: Because the math class story is just persistence, which is a humble brag. The IDOR story is an actual mistake I made that hurt other people and forced me to change how I think about building things. That's a real failure with a real lesson.

INTERVIEWER: Agreed. Don't lead with the math class story in your interviews.

CANDIDATE: Noted. Lead with the one where I actually screwed up.
