---
interview_id: 02_system_design
participant: Anthony Chan (candidate, self-recorded)
role: System design — design a URL shortener
interview_type: technical
date: 2026-04-12
---

INTERVIEWER: Alright, today I'd like you to design a URL shortener. Like bit.ly. Take it from there.

CANDIDATE: Okay cool, so let me think about this. We need a service where someone gives us a long URL and we give them back a short one, and when someone hits the short one we redirect them. I'd start with — okay so we'd need a database to store the mapping from short code to long URL. We'd need an API endpoint to create short codes and another to do the redirect. For the short code generation I'd probably use base62 of a counter or hash the long URL.

INTERVIEWER: Slow down. Before you jump into the design, what questions would you ask about requirements?

CANDIDATE: Oh, right. Um, so I'd ask about scale — how many URLs are we shortening per day, how many redirects, what's the read-to-write ratio. I'd ask about analytics — do we need to track click counts. Custom short codes — can users pick their own. Expiration — do links live forever. Auth — anyone or just signed-in users.

INTERVIEWER: Good. Let's say one billion URLs total in the system, ten thousand creates per day, ten million redirects per day. No custom codes, no expiration, no auth. Click count tracking yes.

CANDIDATE: Okay so reads are way heavier than writes. Like a thousand to one. That means I want the read path to be really cheap. I'd put a cache in front of the database — Redis or something — for the hottest URLs. For the database itself with a billion rows I'd use something with good range read performance... probably DynamoDB or just a sharded SQL database keyed on the short code.

INTERVIEWER: Why not a single SQL database?

CANDIDATE: For a billion rows with ten million reads a day? Honestly maybe I could. Postgres can handle that. I was reaching for sharding too early. Let me back off — start with a single Postgres, add read replicas if needed, only shard when the data outgrows one machine. Premature scaling is a real failure mode.

INTERVIEWER: Good correction. What about the short code generation?

CANDIDATE: A few options. Counter plus base62 encoding gives you the shortest codes but you need a globally unique counter, which is a bottleneck. Random base62 with collision checking works but you get longer codes for the same uniqueness guarantees. Hashing the long URL gives idempotency — the same long URL always gives the same short code — but then you have to handle collisions when two different URLs hash to the same prefix. I'd probably go counter-based with each app server pre-allocating a range of IDs from a coordinator, so you avoid the per-request bottleneck.

INTERVIEWER: How would you track click counts?

CANDIDATE: On every redirect, log a click event. Don't update the row synchronously — fire and forget into a queue or just write to a log, then aggregate offline. The redirect itself should be cache-hit + 302 + done, like five milliseconds.

INTERVIEWER: What about analytics latency? If a user looks at their click count, how recent is it?

CANDIDATE: Depends on the aggregation cadence. Could be near-realtime with Kafka + a stream processor, or batch hourly with a simpler setup. I'd default to the simpler one and only upgrade if customers are complaining.

INTERVIEWER: We're running low on time. Anything you'd want to add?

CANDIDATE: Yeah — I rushed the requirements phase at the start. Next time I'd take a full two or three minutes there even if it feels slow. The design decisions all flow from the requirements and I was already designing before I knew what I was designing for.

INTERVIEWER: That's a fair self-assessment. Thanks.
