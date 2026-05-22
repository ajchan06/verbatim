# Synthetic Interview Dataset — Design Doc

## Why synthetic
Real customer-research transcripts are not publicly available at meaningful scale,
and the few academic datasets (e.g. Enriquez gig-worker corpus) sit in the wrong
domain. We need transcripts that look like the work a *Great Question customer*
would actually do: B2B SaaS user research, JTBD interviews, pricing studies,
churn diagnostics.

Generating them ourselves has a second benefit: we know the ground truth.
Every transcript has planted facts; we use those facts to score retrieval and
faithfulness in our eval harness.

## The study
**Product (fictional):** Linkup — a meeting-scheduling tool that competes with
Calendly. Solo plan is free; Team plan is $12/user/month.

**Research question:** Why do trial users churn? What do power users love?
What's blocking team expansion?

10 interviews, ~15–25 turns each, semi-structured. Mix of:
- 4 churned trial users (Sarah, Marcus, Priya, Tom)
- 3 active power users (Diana, Jamal, Lena)
- 2 team admins evaluating expansion (Rafael, Chloe)
- 1 lost-deal — chose competitor (Yuki)

## Planted ground-truth facts (drives the eval set)

| Fact ID | Claim | Who mentions it |
|---|---|---|
| F1 | Onboarding has too many setup steps before first link can be shared | Sarah, Marcus, Tom |
| F2 | The mobile app is much worse than the web app | Priya, Diana, Yuki |
| F3 | Round-robin routing is the #1 feature users on Team plan love | Diana, Jamal, Rafael |
| F4 | Pricing is felt as too steep for teams under 10 people | Marcus, Chloe, Yuki |
| F5 | Salesforce integration is broken / unreliable | Jamal, Rafael, Chloe |
| F6 | Users want Notion integration (not yet built) | Lena, Diana |
| F7 | Sarah's main blocker was buffer-time defaults | Sarah only |
| F8 | Yuki switched to SavvyCal specifically for the email-embed feature | Yuki only |
| F9 | Tom is a freelancer; Marcus runs a 4-person agency | Tom, Marcus |
| F10 | Chloe's team is 35 people in customer support | Chloe only |

These facts let us build precise evals: e.g. "which interviewees mention
broken Salesforce integration?" has a known answer of {Jamal, Rafael, Chloe}.
