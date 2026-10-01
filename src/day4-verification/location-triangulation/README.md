# Location triangulation

**Day 4 · 60 min · Lab · No GPU (numpy and geometry; seconds of compute)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sg-ai-safety-hub/FAST/blob/main/src/day4-verification/location-triangulation/lab.ipynb)

## Objective

Build delay-based geolocation — landmark servers challenge a chip, time the reply, and turn
round-trip latency into a hard upper bound on distance — and work out what the region it produces
actually establishes. Export controls are written about places; this is the main proposal for
checking one after the crates have shipped.

## What it surfaces

The physics gives a real one-sided guarantee, which is rarer than it sounds: an operator can stall
and look further away, but cannot reply before the signal arrives, so the feasible region always
contains the truth. What it does not give is precision. For a cluster in Johor Bahru, Singapore,
Batam and Kuala Lumpur are all consistent with the same measurements — four cities, three
countries — so the false claim the operator actually wants to make costs them nothing and cannot
be contradicted.

Three limits do the damage, and none of them is the geometry. The measurement locates whatever
answered the challenge, so without a hardware root of trust it says nothing about the chips.
Resolution is set by the nearest landmark and by the jitter floor: a millisecond of jitter adds
100 km to every bound, and the Singapore–Johor border is 0.17 ms wide. And the tempting fix,
assuming a realistic path speed rather than a physical one, tightens the region, excludes the honest
operator, and leaves a plausible-looking answer centred a couple of hundred kilometres from the
truth with nothing in the output to say so.

## Structure

Five parts, four functions you write: the distance bound from a round trip, the intersection of
every landmark's bound over a grid, the delay an operator needs to make a false claim consistent,
and the right estimator for repeated measurements. Then what sets the resolution, what calibration
costs, and the case the mechanism does catch.

Repeated pings are worth the space they take. Jitter is one-sided like the operator's stalling is,
so the estimator is the minimum rather than the mean, and the region converges after about thirty
pings onto the floor set by path stretch and stops there. Repetition removes random error and
leaves systematic error untouched, which lands better as a measured result than as a sentence.

Part 5 is the counterweight to the rest. A cluster diverted to Shenzhen while claiming Singapore is
caught outright — the region is under 100 km across and the claim sits 2,500 km outside it — and
stalling enough to cover the claim inflates the region to 7,000 km, which answers nothing. The
mechanism works on the question export control actually asks; it fails on the one the lab opens
with.

The region is drawn as text and as a map over country borders. When the region is tight the map
frames it and adds an inset magnifying it, because a well-constrained region is a couple of percent
of the frame and would otherwise be an invisible speck — the figure would look empty exactly when
the measurement had worked. Part 4 draws the jitter
floor as a true geodesic circle over the Singapore Strait, against the borders it is supposed to
resolve — it covers three countries.

Borders are [Natural Earth](https://www.naturalearthdata.com/) vectors bundled inside the package,
not map tiles. The lab drew on hosted tiles first and was blocked for it: twenty participants
pulling from volunteer-run servers is what the OpenStreetMap tile usage policy exists to prevent,
and the commercial tile providers want an API key. Natural Earth is public domain, so the data
ships in the wheel and the maps need no tile server, no key, and no network.
`tools/build_borders.py` regenerates them if the extent ever changes.

No model and no GPU, so it slots in wherever the day has room, and it pairs with the recomputation
lab: both mechanisms fail at the same joint, which is that the party being audited authors the
evidence.

## Prerequisites

Day 0 done. Nothing else; the geometry is self-contained. Reading it after the recomputation lab
makes the shared failure mode obvious, but either order works.

## A note on the numbers

Round-trip times are synthesised from great-circle distance with a per-path stretch factor and
jitter, not measured. They sit in the right range for real inter-city latency, and the lab says so
where it matters. The geometry and the one-sidedness are the content; the specific milliseconds
are not.

## On framing

The worked examples in this literature are usually North American or European, where landmark
density is high and the borders that matter are far apart. The Singapore–Johor–Batam triangle is
three countries inside an hour's drive, and a real place people build datacentres. Whether a
verification mechanism is fit for purpose is a question about a specific map.

## Sources

- Brass and Aarne, *Location Verification for AI Chips* (IAPS, 2025) —
  [iaps.ai](https://www.iaps.ai/s/LocationVerificationforAIChips.pdf)
- *AI Compute governance: Verifying AI chip location* —
  [LessWrong](https://www.lesswrong.com/posts/uSSPuttae5GHfsNQL/ai-compute-governance-verifying-ai-chip-location)
- Gueye, Ziviani, Crovella and Fdida, *Constraint-Based Geolocation of Internet Hosts*,
  IEEE/ACM Transactions on Networking 14(6), 2006 — [doi](https://doi.org/10.1109/TNET.2006.886332)
- Petrie et al., *Flexible Hardware-Enabled Guarantees* (2025) — [arXiv:2506.15093](https://arxiv.org/abs/2506.15093)
- Aarne, Fist et al., *Secure, Governable Chips* (CNAS, 2024) —
  [cnas.org](https://www.cnas.org/publications/reports/secure-governable-chips)
