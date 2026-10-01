# Day 4: Verification

How you check a claim about a computation someone else ran, on hardware you don't control,
without being handed the secrets. The day works through what verification mechanisms actually
establish, which is usually narrower than the claim they are asked to support, and why the
serious proposals end up in the hardware rather than in the auditor's statistics.

## Run of show

- [`inference-recomputation/`](inference-recomputation/) · **Inference recomputation** · Lab
- [`location-triangulation/`](location-triangulation/) · **Location triangulation** · Lab

Order here is the source of truth. Directories are unnumbered so reordering the run of show
doesn't rename paths or break bookmarked Colab links. Guest lectures have no directory; speaker
briefs stay in the internal docs, and any reading a session assumes goes in the exercise above it.

## Notes

Both labs are built to fail in a specific way, and they fail at the same joint: the party being
audited authors the evidence. The recomputation lab makes its audit work cleanly first and then
defeats it three times without anyone breaking a hash or outrunning the sampling. The triangulation
lab has a genuine one-sided physical guarantee and still cannot answer the question an export
licence asks. The preconditions they leave standing — the record is produced by something the
operator doesn't control, the verifier chooses the inputs, the measurement resolves finer than the
thing being decided — are what the treaty-verification and hardware-enabled-mechanisms sessions
build on.

Recomputation needs a T4 and about 90 minutes; triangulation needs no GPU and about 45, so it can
move in the running order or absorb a session that overran.

Watch the framing on this day in particular. Arms-control analogies and the AI 2040 material are
useful tools to think with, not a default world view, and this cohort is international with an
Asia focus. Caveat them in the material rather than in the room.
