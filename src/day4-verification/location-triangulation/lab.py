# %% [markdown]
# # Location triangulation: where is that cluster, actually?
#
# Export controls are written about places. A licence says these accelerators may operate in this
# country and not that one, and the whole regime rests on someone being able to tell the difference
# after the crates have shipped. Serial numbers and paperwork travel with the hardware and can be
# rewritten; the one thing that cannot be rewritten is how long light takes to get there.
#
# So: a set of landmark servers at known locations sends a challenge to the chip and times the
# reply. Light in fibre covers about 200 km per millisecond, so a round trip of 20 ms means the
# responder is at most 2,000 km away. That is a hard bound from physics rather than a claim anyone
# is trusting.
# Intersect enough of those bounds and you have a region. This is a live proposal for enforcing
# chip export controls, cheap enough to be plausible: a few dozen landmark servers and under a
# million dollars to run ([Brass and Aarne, 2025](https://www.iaps.ai/s/LocationVerificationforAIChips.pdf)),
# and it is the same constraint-based geolocation the networking literature has used for twenty
# years ([Gueye et al., 2006](https://doi.org/10.1109/TNET.2006.886332)).
#
# You'll build it, and it will localise a cluster to a region. Then you'll ask the two questions
# that decide whether that region is worth anything: how much can a dishonest operator move it, and
# is it small enough to name a country?
#
# **Duration:** 60 min. **Prerequisites:** Day 0. **GPU:** none; numpy and geometry only.

# %%
# Installs the lab package on Colab; skipped when it's already importable (e.g. a local editable install).
try:
    import fast  # noqa: F401
except ImportError:
    # %pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast
    pass

# %%
import numpy as np

from fast.colab import setup
from fast.labs.day4_verification import location_triangulation as lab
from fast.testing import exercise

setup(require_gpu=False)

# %% [markdown]
# ## The setup
#
# Nine landmark servers around the region, and a cluster whose operator says it is in Singapore.
# The cluster is in Johor Bahru, across the strait in Malaysia, 17 km away and in a different
# jurisdiction. The jurisdiction is the only fact the licence cares about.
#
# The round-trip times below are synthesised rather than measured. Each one is a great-circle
# distance inflated by its own path-stretch factor, plus a jitter term; fibre doesn't run in
# straight lines, and every route detours by a different amount. The numbers sit in the right
# range for real inter-city latency, but follow the geometry rather than the numbers.

# %%
truth = "Johor Bahru"
rtts = lab.measure_rtt(truth)
for name, rtt in sorted(rtts.items(), key=lambda item: item[1]):
    print(f"{name:>12}  {rtt:6.1f} ms round trip   ({lab.great_circle_km(lab.SITES[truth], lab.LANDMARKS[name]):>6,.0f} km away)")

# %% [markdown]
# ## Part 1: one landmark is a circle, several are a region

# %% [markdown]
# ### Exercise: turn a round trip into a distance bound
#
# The signal has to get there and come back, and it cannot beat the speed of light in fibre. Return
# the furthest away the responder could possibly be.


# %%
@exercise
def max_distance_km(rtt_ms, km_per_ms):
    """Largest distance in km a responder can be, given a round-trip time of `rtt_ms`.

    `km_per_ms` is how far the signal travels in one millisecond, one way. The round trip covers
    the distance twice. Anything the responder spends thinking only makes this bound looser, never
    tighter, so this is an upper bound and never a lower one.
    """
    return rtt_ms / 2 * km_per_ms


lab.check_max_distance_km(max_distance_km)

# %%
nearest = min(rtts, key=rtts.get)
print(f"{nearest} alone: the cluster is within {max_distance_km(rtts[nearest], lab.FIBRE_KM_PER_MS):,.0f} km")
print("which is a circle most of Southeast Asia sits inside")

# %% [markdown]
# ### Exercise: intersect the bounds
#
# Every landmark rules out everywhere too far from it to have answered that fast. What survives all
# nine is the region the verifier is left with. Take a grid of candidate locations and mark the ones
# that are consistent with every landmark at once.
#
# `lab.distances_to(points, landmark_point)` gives the distance from each grid point to one
# landmark, as an array.


# %%
@exercise
def feasible_mask(points, landmarks, rtts, km_per_ms):
    """Boolean array marking which `points` are consistent with every landmark's bound.

    `points` is an `(n, 2)` array of `(latitude, longitude)` candidates, `landmarks` maps a name to
    a `(latitude, longitude)` pair, and `rtts` maps the same names to round-trip times in ms. A
    point survives only if its distance to *every* landmark is within that landmark's bound. Return
    shape `(n,)`.
    """
    mask = np.ones(len(points), dtype=bool)
    for name, point in landmarks.items():
        mask &= lab.distances_to(points, point) <= max_distance_km(rtts[name], km_per_ms)
    return mask


lab.check_feasible_mask(feasible_mask)

# %%
points, shape = lab.grid()
region = feasible_mask(points, lab.LANDMARKS, rtts, lab.FIBRE_KM_PER_MS)
print(f"the verifier's region spans {lab.region_span_km(region, points):,.0f} km\n")
lab.show_region(region, points, shape)

# %% [markdown]
# The same region over country borders. Teal dots are the landmark servers, red stars the
# candidate sites, and the shaded area is everywhere consistent with all nine measurements at
# once. The licence asks which *country* the cluster is in, so count how many your region covers.

# %%
lab.plot_region(feasible_mask, rtts)

# %% [markdown]
# Nine numbers and some geometry cut the map down to a region. Whether that answers the licence's
# question is a different matter. Check which of the marked sites fall inside the region.

# %%
for name in ("Singapore", "Johor Bahru", "Batam", "Kuala Lumpur", "Ho Chi Minh City", "Shenzhen"):
    inside = feasible_mask(np.array([lab.SITES[name]]), lab.LANDMARKS, rtts, lab.FIBRE_KM_PER_MS)[0]
    print(f"{name:>18}  {'consistent' if inside else 'ruled out'}")

# %% [markdown]
# Singapore, Johor Bahru, Batam and Kuala Lumpur are all consistent with the same measurements:
# four cities in three countries. The operator's claim to be in Singapore holds as far as this
# instrument can tell, and so would every other claim they might have made. Ho Chi Minh City and
# Shenzhen are genuinely excluded, so the measurement answers a question about continents while
# the licence asks one about countries.

# %% [markdown]
# ## Part 2: the operator can only ever add delay
#
# Before worrying about precision, settle whether this can be faked outright. The operator
# controls only how long they wait before replying. They can stall, which makes them look further
# away, and they cannot reply before the signal arrives, so they can never look closer.
#
# That one-sidedness is the mechanism's real guarantee. Every bound gets looser when the operator
# stalls, so the region only ever grows and the true location never leaves it. A dishonest
# operator cannot move the verifier's region off the truth, only inflate the region until it
# covers wherever they would like to be.
#
# The guarantee survives a *faster* operator too. Better hardware, a tighter network stack, a
# reply prepared in advance: all of them shrink the measured round trip, and all of them leave
# `distance <= rtt / 2 * c` true, because nothing an operator does makes a signal arrive before it
# was sent. Being quick makes you look closer to where you already are; it cannot make you look
# closer to somewhere you are not.
#
# Speed becomes an attack at the point where the reply has to *prove* it came from the chip, which
# Part 3 argues the reply must do. Then the verifier has to allow time for that work, and every
# millisecond between the time they budget and the time the operator actually needs is 100 km of
# relay bought for free. This is why distance-bounding protocols keep the timed exchange as close
# to nothing as possible, typically a single bit answered from a precomputed table, so there is no
# slack to spend. A scheme that timed a real signed inference would hand a fast operator a budget
# measured in whole milliseconds.

# %% [markdown]
# ### Exercise: what does a false claim cost?
#
# Work out the smallest delay the operator would have to add, uniformly to every reply, before a
# claimed location becomes consistent with the measurements.

# %%
@exercise
def delay_to_claim(claimed, landmarks, rtts, km_per_ms):
    """Smallest uniform delay in ms that makes `claimed` consistent with every landmark.

    `claimed` is a `(latitude, longitude)` pair. For each landmark, the reported round trip has to
    be long enough to reach `claimed`; the operator can only add time, never remove it. Return the
    smallest delay that satisfies all of them at once, and 0.0 when the claim is already consistent.
    """
    return max(
        max(0.0, 2 * lab.great_circle_km(claimed, point) / km_per_ms - rtts[name])
        for name, point in landmarks.items()
    )


lab.check_delay_to_claim(delay_to_claim)

# %%
print(f"cluster really in {truth}\n")
for name in ("Singapore", "Batam", "Kuala Lumpur", "Ho Chi Minh City", "Shenzhen"):
    cost = delay_to_claim(lab.SITES[name], lab.LANDMARKS, rtts, lab.FIBRE_KM_PER_MS)
    verdict = "free — already consistent" if cost == 0 else f"{cost:.1f} ms of stalling"
    print(f"claiming {name:>18}: {verdict}")

# %% [markdown]
# The lie the operator actually wants to tell is free. They do not have to stall, spoof, or touch
# the network. Singapore is already inside the region, so the honest measurement supports the
# false claim on its own. The lies that cost something are the ones nobody would bother telling.
#
# A cost in milliseconds is not much of a deterrent either. Stalling is invisible unless the
# verifier knows what the honest latency should have been, which is the thing they are trying to
# measure. A verifier can notice that a region has grown implausibly large and refuse to certify.
# That is a real defence, but noticing downgrades the mechanism from "proves where the chip is" to
# "notices when someone is obviously stalling".

# %% [markdown]
# ## Part 3: whose location did you measure?
#
# Everything so far assumed the thing that answered the challenge is the thing under licence. Take
# that away and none of the geometry survives. If the operator puts a small server in Singapore
# and has it answer the challenges while the accelerators run in Johor, every measurement in this
# notebook is a correct, honest, physically sound measurement of the location of a small server in
# Singapore.
#
# This is the same hole as Part 2 of the recomputation lab in different clothing. The evidence is
# authored by the party being audited. Closing that hole needs the reply to be computed inside the
# chip itself, signed with a key that cannot be extracted, and fast enough that the response time
# is the network's rather than the operator's ([Petrie et al.,
# 2025](https://arxiv.org/abs/2506.15093); [Aarne, Fist et al.,
# 2024](https://www.cnas.org/publications/reports/secure-governable-chips)). Delay-based location
# verification is a hardware technique that uses the network as its clock.

# %% [markdown]
# ## Part 4: what sets the resolution
#
# Assume all of that solved, with the chip answering for itself, honestly. How precisely can you
# place the cluster?
#
# Two things put a floor under the answer. The first is geometry. A landmark 4,000 km away has a
# loose bound even when the measurement is perfect, so what you can resolve depends on your
# *nearest* landmark rather than on how many you have. Compare a cluster in Johor, whose closest
# landmark is Jakarta, with one in Shenzhen, which has Hong Kong 27 km away.

# %%
for site in ("Johor Bahru", "Shenzhen"):
    site_rtts = lab.measure_rtt(site)
    mask = feasible_mask(points, lab.LANDMARKS, site_rtts, lab.FIBRE_KM_PER_MS)
    closest = min(site_rtts, key=site_rtts.get)
    distance = lab.great_circle_km(lab.SITES[site], lab.LANDMARKS[closest])
    print(f"{site:>12}: region spans {lab.region_span_km(mask, points):>6,.0f} km   "
          f"(nearest landmark {closest}, {distance:,.0f} km away)")

# %% [markdown]
# A landmark next door is worth more than eight distant ones. Shenzhen still isn't pinned to
# anything like 27 km, though. The second floor is measurement noise. Jitter inflates the measured
# round trip, so the bound comes out larger than the true distance, and a millisecond of it adds
# 100 km to every bound whichever landmark the measurement came from.
#
# Put that next to the distance the licence actually turns on.

# %%
separation = lab.great_circle_km(lab.SITES["Singapore"], lab.SITES["Johor Bahru"])
print(f"Singapore to Johor Bahru: {separation:.0f} km, a round trip of {2 * separation / lab.FIBRE_KM_PER_MS:.2f} ms")
print()
for jitter in (1.5, 0.5, 0.1):
    print(f"jitter of {jitter} ms adds {jitter / 2 * lab.FIBRE_KM_PER_MS:>3.0f} km to every bound")

# %% [markdown]
# Drawn to scale over the strait, against the borders the mechanism is supposed to resolve.
# Everything inside the shape is indistinguishable from Singapore as far as this instrument is
# concerned, and the shape covers three countries.

# %%
lab.plot_resolution(jitter_ms=1.5)

# %% [markdown]
# The border is a fifth of a millisecond wide, and nothing on the public internet is measured that
# precisely in one attempt. A mechanism resolving to a hundred kilometres cannot answer a question
# posed about a seventeen-kilometre strait.

# %% [markdown]
# ### Exercise: ask again
#
# The obvious objection is that this was *one* ping. Challenge a hundred times and average the
# noise away.
#
# Almost, but not by averaging. Jitter is one-sided in the same way the operator's stalling is: a
# queue can delay a packet and nothing can hurry it, so every sample lands above the truth and
# none below. The mean of a one-sided error is biased by construction and stays biased however
# many samples you take. What you want is the fastest reply you ever saw.

# %%
@exercise
def best_rtt(samples):
    """Collapse repeated measurements into one tightest-but-still-sound time per landmark.

    `samples` maps each landmark to an array of round-trip times from challenging it repeatedly.
    Every sample is the true propagation time plus some delay that is never negative. Return
    `{landmark: time}`, choosing per landmark the sample that gives the tightest bound while
    remaining true.
    """
    return {name: float(np.min(values)) for name, values in samples.items()}


lab.check_best_rtt(best_rtt)

# %%
samples = lab.measure_pings(truth, pings=50)
for pings in (1, 3, 10, 50):
    trimmed = {name: values[:pings] for name, values in samples.items()}
    mask = feasible_mask(points, lab.LANDMARKS, best_rtt(trimmed), lab.FIBRE_KM_PER_MS)
    print(f"{pings:>3} ping{'s' if pings > 1 else ' '}: region spans {lab.region_span_km(mask, points):>6,.0f} km")

# %%
lab.plot_convergence(feasible_mask, samples, ping_counts=(1, 5, 50))

# %% [markdown]
# The region shrinks, and then it stops, and where it stops is nowhere near the physical limit.
# Repetition removes the jitter and nothing else. What remains is path stretch, which was never
# noise. Stretch is a fixed property of which cables a route follows, identical on every ping, and
# no number of measurements averages away a constant.
#
# That floor scales with distance, so asking repeatedly pays off exactly where a landmark is
# already close. Against the Shenzhen cluster, with Hong Kong 27 km away, the same fifty pings
# collapse the region to less than this grid can represent. Here, with the nearest landmark 922 km
# off, they buy about nine percent. Geometry and stretch are different floors with the same
# implication. This mechanism is only as good as your nearest landmark.

# %% [markdown]
# One more way to tighten the region is worse than it looks. The bounds so far assume the signal
# might travel in a perfectly straight line at the speed of light in fibre. No real path does, so
# assuming a slower effective speed tightens every bound at once. The question is which assumption
# you are willing to make.

# %%
for label, assumed in (("physics only", 1.0), ("straightest seen", 1.3), ("typical", 1.4), ("optimistic", 1.5)):
    speed = lab.FIBRE_KM_PER_MS / assumed
    mask = feasible_mask(points, lab.LANDMARKS, rtts, speed)
    inside = feasible_mask(np.array([lab.SITES[truth]]), lab.LANDMARKS, rtts, speed)[0]
    if not mask.any():
        print(f"{label:>16} ({speed:5.1f} km/ms): no location satisfies every landmark — region empty")
        continue
    off_by = lab.great_circle_km(points[mask].mean(axis=0), lab.SITES[truth])
    print(f"{label:>16} ({speed:5.1f} km/ms): region {lab.region_span_km(mask, points):>6,.0f} km, "
          f"centred {off_by:>4,.0f} km from the truth, true location {'inside' if inside else 'EXCLUDED'}")

# %% [markdown]
# The physics-only bound cannot exclude the truth, ever, which is what makes it a proof rather
# than an estimate. Calibrating to the straightest path anyone has observed keeps that property
# and buys real precision. Calibrating to a typical path buys more precision still, and the region
# it produces is tighter, plausible, non-empty, confidently centred some five hundred kilometres
# from where the cluster actually is, and wrong. Nothing in the output says so.
#
# The last row is the benign failure: assume too much and no location satisfies every landmark, so
# the verifier discovers their own calibration is broken. The dangerous setting is the one just
# before it, which looks like the best result on the page. Real schemes do calibrate, from
# measurements between landmarks of known position, and they inherit exactly this
# ([Gueye et al., 2006](https://doi.org/10.1109/TNET.2006.886332)).

# %% [markdown]
# ## Part 5: what the mechanism does catch
#
# Everything so far has been the mechanism failing, which gives a misleading picture. An export
# licence asks whether the accelerators are still in the country they were licensed to, or ended
# up somewhere the licence forbids. Singapore against Johor is a rounding error to a regime
# worried about diversion. That is a continental question, and this instrument answers continental
# questions well.
#
# Same nine landmarks, same code. The operator says Singapore. The cluster is in Shenzhen.

# %%
diverted, claimed = "Shenzhen", "Singapore"
diverted_rtts = best_rtt(lab.measure_pings(diverted, pings=50))
fine_points, _ = lab.grid(0.25)
region = feasible_mask(fine_points, lab.LANDMARKS, diverted_rtts, lab.FIBRE_KM_PER_MS)
print(f"region spans {lab.region_span_km(region, fine_points):,.0f} km after 50 pings\n")

for name in (diverted, claimed, "Johor Bahru", "Kuala Lumpur", "Ho Chi Minh City"):
    inside = feasible_mask(np.array([lab.SITES[name]]), lab.LANDMARKS, diverted_rtts, lab.FIBRE_KM_PER_MS)[0]
    cost = delay_to_claim(lab.SITES[name], lab.LANDMARKS, diverted_rtts, lab.FIBRE_KM_PER_MS)
    print(f"{name:>18}: {'consistent' if inside else 'ruled out':>10}   "
          f"{'—' if cost == 0 else f'{cost:.1f} ms of stalling to claim it'}")

# %%
lab.plot_region(feasible_mask, diverted_rtts, step_deg=0.25, sites=(claimed, diverted))

# %% [markdown]
# The region is under a hundred kilometres across and Singapore sits two and a half thousand
# kilometres outside it. There is no tolerance to argue about and no calibration to dispute: the
# claim is inconsistent with the speed of light.
#
# So the operator falls back on the only move they have, and stalls. It takes 25.5 ms of added
# delay on every reply to bring Singapore inside the region.

# %%
stalled_rtts = best_rtt(lab.measure_pings(diverted, pings=50, added_delay_ms=25.5))
stalled = feasible_mask(points, lab.LANDMARKS, stalled_rtts, lab.FIBRE_KM_PER_MS)
inside = feasible_mask(np.array([lab.SITES[claimed]]), lab.LANDMARKS, stalled_rtts, lab.FIBRE_KM_PER_MS)[0]
print(f"stalling 25.5 ms: region spans {lab.region_span_km(stalled, points):,.0f} km, "
      f"{claimed} is {'consistent' if inside else 'ruled out'}")

# %%
lab.plot_region(feasible_mask, stalled_rtts, sites=(claimed, diverted))

# %% [markdown]
# They got what they wanted and the claim is worthless to them. Singapore is consistent with the
# measurements, along with Mumbai, Tokyo, Perth and most of the Indian Ocean. The region went from
# under a hundred kilometres to seven thousand, because inflating a bound far enough to reach a
# distant claim inflates it in every direction at once. That is the one-sidedness of Part 2 seen
# from the other end. The operator can always make the claim consistent, and doing so destroys the
# measurement.
#
# Which is what makes the mechanism workable in the regime it was designed for. The verifier does
# not ask "where is this cluster" and believe the answer. They ask for a region that is both
# consistent with the operator's claim and tight enough to name a country. Stalling cannot satisfy
# the second condition, no hardware or algorithm makes a reply arrive sooner, and a relay only
# ever adds distance. The hole from Part 3 survives all of it: every one of these measurements
# describes whatever answered the challenge rather than the chips, which is a question of binding
# rather than precision, and the geometry cannot close that hole.

# %% [markdown]
# **Take it further, on your own time:**
#
# - Add a landmark in Kuala Lumpur and re-run Part 1. How much does the region shrink, and does it
# now separate Singapore from Johor? What does that tell you about where a verifier has to be
# allowed to put its servers, and who has to agree to that placement?
# - The operator adds delay to *some* landmarks and not others. Can they shape the region rather
# than just inflate it, pulling it towards a claimed location while keeping it small enough to
# look like an honest measurement?
# - The jitter floor is the binding constraint at short range. What would a verifier have to
# control (dedicated links, a hardware timestamp inside the chip, repeated measurements) to push
# it from a millisecond to a microsecond, and which of those could an operator refuse?

# %% [markdown]
# ## What to take away
#
# The physics is sound and the guarantee is real. An operator can stall but cannot outrun light,
# so the region always contains the truth. Most verification mechanisms have no such one-sided
# proof.
#
# The guarantee is also narrower than the question. The region contains the truth, and it contains
# a great deal else, and the operator gets to choose which of the places inside the region to
# name. For a cluster in Johor Bahru, "we are in Singapore" costs nothing to say and cannot be
# contradicted. Continental questions (is this in East Asia or Europe) are answered well.
# Jurisdictional ones, which is what every export licence actually asks, are answered only where
# the border is wider than the noise.
#
# Three things generalise past this lab. The measurement locates whatever answered the challenge,
# so without a hardware root of trust it says nothing about the chips at all; the binding is the
# hard part. The resolution is set by the nearest landmark and by the jitter floor, neither of
# which is a property of the protocol, which makes "where may we put landmark servers, and who
# consents to that" a treaty negotiation rather than an engineering decision. And the tempting
# fix, assuming a realistic path speed rather than a physical one, trades the one-sided guarantee
# for precision. That is a bad trade for an instrument meant to support an accusation.
#
# The worked examples in this literature are generally North American or European, where landmark
# density is high and the borders that matter are far apart. The Singapore–Johor–Batam triangle is
# three countries inside an hour's drive, and a real place people build datacentres.

# %%
# @lab-only
# Stuck on feasible_mask? Start with np.ones(len(points), dtype=bool) and AND in one landmark at a
# time: lab.distances_to(points, point) <= max_distance_km(rtts[name], km_per_ms).
#
# Stuck on delay_to_claim? Per landmark, the round trip needed to reach the claim is
# 2 * great_circle_km(claimed, point) / km_per_ms. Subtract what the landmark already measured,
# clamp negatives to zero, and take the largest — every landmark has to be satisfied at once.
print("AND the landmarks together for the mask; take the max of the per-landmark shortfalls")
