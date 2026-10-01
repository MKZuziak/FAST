# Inference recomputation

**Day 4 · 90 min · Lab · T4 GPU (~3 min of compute, all short forward passes — nothing trains)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sg-ai-safety-hub/FAST/blob/main/src/day4-verification/inference-recomputation/lab.ipynb)

## Objective

Audit a claim about a computation you didn't run. A datacentre is under an agreement to run one
particular model; you get their log, recompute a random sample of it on your own copy, and check
the numbers match. Build that auditor, watch it work, then defeat it three times from the
operator's side.

The same audit serves three parties who rarely turn up in the same room. A treaty inspector
confirming a signatory runs the system it declared. A safety team confirming the model answering
requests is the one that went through evals, rather than a quantised variant swapped in for
throughput. And a buyer of inference through a reseller confirming they get the model they pay
for rather than a cheaper one behind the same API name. The lab is written around the first,
since that is the Day 4 theme, but the mechanism and every one of its failures are identical for
the other two.

## What it surfaces

Recomputation is the cheapest verification mechanism there is — no new hardware, no cryptography,
no trust in the operator — and it runs through most of the near-term compute verification
literature. Built naively it works: exact digest matches, a swapped model caught instantly, and
sampling arithmetic favourable enough that a one-in-ten-thousand cheat falls to recomputing a
third of a percent of a day's traffic.

Part 1 also shows the audit catching a swapped model outright, which is where it earns the
comparison with behavioural testing. The substitutions are named rather than synthetic: the
agreement is about Qwen2.5-0.5B-Instruct, and the operator is caught serving Qwen2.5-0.5B (the
base model it was tuned from, already on the same disk), then the promised weights rounded to
int8, then to int4. Quantisation is the cheat with the clearest commercial motive, since it needs
no second checkpoint and the answers stay indistinguishable to a reader. All three fail on the
first row audited, because the audit never asks what the model does. A backdoored checkpoint is behaviourally invisible until
somebody sends the trigger, but it is still a weight change, so it fails an exact digest on
perfectly ordinary prompts while the backdoor sleeps. Recomputation does not test behaviour, and
that is its one clear advantage over the Day 3 probes.

Then each cheat removes one unstated assumption. The operator authors the log, so recomputation
checks a document against a model and never touches the serving path. The operator chooses which
rows to cheat on and can recognise audit traffic, so a sample drawn from their own log is one
they can anticipate. And the auditor's arithmetic is not the operator's, so exact equality decays
into a tolerance wide enough to hide a real weight modification inside.

What's left is a list of preconditions that are all hardware or protocol requirements rather than
better statistics — which is the argument for pushing verification down into the chip, arrived at
by failing without it rather than asserted up front.

## Structure

Four parts. Log a day of synthetic traffic and spot-check it under the most favourable conditions
that exist (same weights, same precision, same machine), getting bitwise-exact matches. Then the
three cheats: an honest log over dishonest serving, a 2% cheat rate against a random sample, and
the tolerance that appears the moment the auditor moves to their own hardware. Four functions you
write are checked in the notebook: the spot check, the sample size, the tolerance, and what the
tolerance buys.

The maths is sampling and quantiles, so the security half of the room is on home ground with
inspection regimes and false-positive budgets; the ML half gets a concrete demonstration of why
floating-point non-associativity is a governance problem and not a footnote. Part 4 depends on
reduced-precision recomputation genuinely diverging, which it does: the honest disagreement is
larger than a weight tamper big enough to change the model's answers.

## Prerequisites

Day 0 done. Day 3's backdoor lab is a useful reference point for Part 3 — the same mechanism seen
from the auditor's side, as something to find rather than something a probe catches — but the
notebook doesn't assume it.

## A note on scope

The claim under audit is model identity — "we ran this model on this workload" — not model safety,
so the lab needs no harmful prompts and ships none. The workload is synthetic ordinary traffic
generated at runtime, and the tampered weights exist only inside a context manager that restores
them on exit. Nothing weaponizable is committed.

## Sources

- Cankaya, *A system overview for near-term, low-trust AI compute verification* (2026) —
  [LessWrong](https://www.lesswrong.com/posts/fgvmKqRGvBteKeDoc/a-system-overview-for-near-term-low-trust-ai-compute)
- Shavit, *What does it take to catch a Chinchilla?* (2023) — [arXiv:2303.11341](https://arxiv.org/abs/2303.11341)
- Aarne, Fist et al., *Secure, Governable Chips* (CNAS, 2024) —
  [cnas.org](https://www.cnas.org/publications/reports/secure-governable-chips)
- Petrie et al., *Flexible Hardware-Enabled Guarantees* (2025) — [arXiv:2506.15093](https://arxiv.org/abs/2506.15093)
- Schabl et al., *Attestable Audits* (2025) — [arXiv:2506.23706](https://arxiv.org/abs/2506.23706)
- Thinking Machines, *Defeating Nondeterminism in LLM Inference* (2025) —
  [thinkingmachines.ai](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
