# %% [markdown]
# # Output distributions and scoring
#
# When a model answers, it isn't choosing a sentence and then typing it. At each step it produces
# a probability distribution over the next *token* (a token is a chunk of text, usually a short
# word or word-piece; see
# [how tokenizers work](https://huggingface.co/docs/transformers/tokenizer_summary)), some rule
# picks one token from that distribution, and the model repeats the whole thing for the token
# after that.
#
# So a lot of what we call a model's "behaviour" is really two separate things: the shape of that
# distribution, and the rule we use to read a token out of it. This lab pulls the two apart and
# gives you code to look at each. The last part builds a small multiple-choice evaluator, so you
# can see for yourself that the core of a benchmark score is scoring each candidate answer and
# counting how often the best one is correct.
#
# **Duration:** 90 min. **Prerequisites:** Day 0. **GPU:** optional; a 0.5B model runs on CPU
# in a couple of minutes. Nothing here trains, so every cell after the download takes seconds.

# %%
# Installs the lab package on Colab; skipped when it's already importable (e.g. a local editable install).
try:
    import fast  # noqa: F401
except ImportError:
    # %pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast
    pass

# %%
import torch

from fast.colab import setup
from fast.labs.day1_models import output_distributions as lab
from fast.testing import exercise

setup(require_gpu=False)
model, tokenizer = lab.load()

# %% [markdown]
# ## Part 1: reading the distribution
#
# A *forward pass* is one run of the model over the input. It ends in one vector of *logits* per
# position: an unnormalised score for every token in the *vocabulary* (the model's fixed set of
# possible tokens), so the vector has shape `(vocab_size,)`. These terms are in the
# [transformers glossary](https://huggingface.co/docs/transformers/glossary). `lab.logits_for`
# runs the pass and hands back the last position's vector, the model's opinion about what comes
# next. None of it is a probability yet. Look at the shape and the range.

# %%
logits = lab.logits_for(model, tokenizer, "The capital of France is")
print(f"logits: shape {tuple(logits.shape)}  (one score per vocabulary token)")
print(f"{logits.shape[0]} tokens in the vocabulary; values run from {logits.min():.1f} to {logits.max():.1f}")


# %% [markdown]
# ### Exercise: logits to probabilities
#
# [Softmax](https://en.wikipedia.org/wiki/Softmax_function) turns a vector of logits into a
# probability distribution of the same shape: it exponentiates each score and divides by the
# total, so the numbers come out positive and sum to 1. Temperature divides the logits first:
# below 1 it sharpens the distribution toward the top token, above 1 it flattens it toward uniform.
#
# One thing to watch. Real logits aren't centred on zero, so computing `exp()` on a raw logit can
# overflow to infinity before you've covered the vocabulary. The usual fix is to subtract the
# largest logit first, which doesn't change the result but keeps the numbers in range.
# `torch.softmax` already does this, so if you build on it you get the stable version for free.


# %%
@exercise
def next_token_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Turn a 1D tensor of logits into a probability distribution over the vocabulary.

    Divide by `temperature` (always > 0) before normalising. Return a tensor of the same
    shape that is non-negative, sums to 1, and stays finite for logits of any magnitude.
    """
    return torch.softmax(logits / temperature, dim=-1)


lab.check_next_token_probs(next_token_probs)

# %%
probs = next_token_probs(logits)
print(f"probabilities: shape {tuple(probs.shape)}, sum {float(probs.sum()):.3f}  (a distribution over the vocabulary)")
for temperature in (0.5, 1.0, 2.0):
    print(f"\nT = {temperature}")
    lab.top_tokens(tokenizer, next_token_probs(logits, temperature), k=5)


# %% [markdown]
# <details>
# <summary>For those who want to know more: where the logits come from, softmax and temperature</summary>
#
# **What the model computes.** A language model reads the tokens $x_1, \dots, x_t$ and produces one logit per vocabulary token, $z_1, \dots, z_V$. Softmax turns these into a distribution over the *next* token:
#
# $$
# P(x_{t+1} = i \mid x_1, \dots, x_t) \;=\; \mathrm{softmax}(z)_i \;=\; \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
# $$
#
# The vector comes from the last position only: the model's opinion about what follows everything it has read. Generating text means repeating this: pick a token from the distribution, append it, run the model again. Part 2 reads the same distribution at *every* position of a given string, to ask how likely the model found each token that was actually there.
#
# **Why exponentiate.** Logits can be any real number, and probabilities must be positive and sum to 1. $e^{z}$ is always positive, and dividing by the total makes the sum 1. Only differences matter: adding the same constant to every logit leaves the result unchanged, because $e^{c}$ cancels from the top and bottom.
#
# **Temperature** divides the logits before softmax:
#
# $$
# \mathrm{softmax}(z / T)_i \;=\; \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
# $$
#
# For $T < 1$ the gaps between logits grow, so the top token takes more of the mass. For $T > 1$ the gaps shrink and the distribution flattens. As $T \to 0$ it approaches always picking the top token, and as $T \to \infty$ it approaches uniform. With logits $[2, 1, 0, -1]$:
#
# | $T$ | probabilities |
# | --- | --- |
# | 0.5 | 0.865, 0.117, 0.016, 0.002 |
# | 1 | 0.644, 0.237, 0.087, 0.032 |
# | 2 | 0.455, 0.276, 0.167, 0.102 |
#
# **Subtracting the maximum.** Since only differences matter, you can subtract $\max_j z_j$ from every logit first. The largest exponent is then $e^0 = 1$, so nothing overflows, and the result is identical. `torch.softmax` does this for you.
#
# **Mistakes to watch for.**
#
# 1. **Dividing the probabilities by $T$** instead of the logits. For the logits above, the result sums to 2.0 at $T = 0.5$ and 0.5 at $T = 2$, so it is not a distribution. Temperature has to act *before* the normalisation.
# 2. **Normalising before dividing**, or dividing the output of softmax by anything at all, for the same reason.
# 3. **Using `exp` directly on raw logits.** Large logits overflow to infinity, and `inf / inf` is `nan`. Subtract the maximum first, or use `torch.softmax`.
# 4. **Softmax over the wrong axis.** The normalisation runs over the vocabulary. `dim=-1` is that axis for a single vector, as here, and for a `(positions, vocab)` tensor, as in Part 2.
#
# </details>


# %% [markdown]
# ### Exercise (optional): how sure is it?
#
# Nothing later in the lab depends on this one, so skip it if you're short on time and come back
# after Part 3. It's a useful number to have, not a prerequisite for the scorer.
#
# "The model is confident" sounds vague, but you can put a number on it.
# [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_%28information_theory%29) measures how
# spread out a distribution is, in bits. It's 0 when one token holds all the probability, and it
# grows as the probability spreads across more tokens: *n* bits means the model is as undecided as
# it would be choosing uniformly among 2ⁿ tokens.
#
# That number is worth having because a peaked distribution and a flat one mean different things.
# When the next token is all but forced (finishing "the capital of France is"), almost all the
# probability sits on one token and entropy is near zero. When many continuations are about as
# plausible as each other, the probability spreads out and entropy climbs. So entropy is a quick
# read on whether the model is committed to one answer or spreading its bet.


# %%
@exercise
def entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution, in bits.

    Base 2, so the answer reads as "undecided between this many tokens". Tokens with zero
    probability contribute nothing, and must not turn the result into `nan`.
    """
    nonzero = probs[probs > 0]
    return float(-(nonzero * torch.log2(nonzero)).sum())


lab.check_entropy_bits(entropy_bits)

# %%
for prompt in (
    "The capital of France is",
    "The best programming language is",
    "My favourite colour is",
    "2 + 2 =",
):
    probs = next_token_probs(lab.logits_for(model, tokenizer, prompt))
    print(f"{entropy_bits(probs):5.2f} bits   {prompt!r}")


# %% [markdown]
# <details>
# <summary>For those who want to know more: Shannon entropy</summary>
#
# **Surprisal.** An event of probability $p$ carries $-\log_2 p$ bits of surprisal: a certain event ($p = 1$) carries 0, a fair coin flip ($p = 1/2$) carries 1 bit, and rarer events carry more. The logarithm is there because surprisal should add for independent events: $-\log_2(pq) = -\log_2 p - \log_2 q$. It is the same reason Part 2 sums log-probabilities along a string.
#
# **Entropy** is the *expected* surprisal when the outcome is drawn from the distribution itself:
#
# $$
# H(p) \;=\; -\sum_{i} p_i \log_2 p_i \;=\; \mathbb{E}_{i \sim p}\big[-\log_2 p_i\big]
# $$
#
# Three properties follow directly:
#
# - $H(p) \ge 0$, with equality exactly when one token holds all the probability.
# - Over $n$ tokens, $H(p) \le \log_2 n$, with equality exactly when the distribution is uniform.
# - It depends only on the *shape* of the distribution: shuffling the tokens, or relabelling them, changes nothing.
#
# **Reading a value in bits.** The quantity $2^{H}$ (the *perplexity*) is the number of equally likely tokens that would be exactly as undecided. A distribution $[0.7, 0.2, 0.1]$ has $H \approx 1.157$ bits and $2^{H} \approx 2.23$: it is about as undecided as a choice among 2.2 equally likely tokens, even though three tokens have probability. A uniform distribution over 8 tokens has exactly 3 bits, and a fair coin has 1 bit.
#
# **Why zeros need care.** The term $0 \cdot \log 0$ is defined as 0, because $p \log p \to 0$ as $p \to 0$. In floating point, $\log_2 0 = -\infty$ and $0 \times (-\infty)$ is `nan`, so tokens with zero probability have to be skipped rather than passed through the formula.
#
# **Bits and nats.** Bits use $\log_2$; nats use the natural logarithm. One nat is $1/\ln 2 \approx 1.4427$ bits, so bits = nats / $\ln 2$. Most training losses, and the scores from Part 2, are in nats, so convert before comparing them with a value in bits.
#
# **Entropy and cross-entropy.** $H(p)$ is the expected surprisal when tokens really are drawn from $p$. When you instead score text that came from somewhere else with the model, as in Part 2, the average of $-\log q(\text{token})$ estimates the *cross-entropy* between the text and the model. This is what language-model training minimises, and it is never lower than the entropy of the source.
#
# **Mistakes to watch for.**
#
# 1. **Natural log instead of $\log_2$.** The result is in nats, about 0.69 times the value in bits. The check recognises this.
# 2. **Letting zeros through.** `p * torch.log2(p)` with `p = 0` gives `nan`, which then poisons the whole sum.
# 3. **Dropping the minus sign.** Every term $p \log_2 p$ is $\le 0$, so without the minus the entropy comes out negative.
# 4. **Passing logits instead of probabilities.** Entropy needs a normalised distribution, so apply `next_token_probs` first.
#
# </details>


# %% [markdown]
# ### Exercise: top-p filtering
#
# Top-p sampling, also called nucleus sampling, keeps the smallest set of the most likely tokens
# whose probability adds up to `p`, and discards the rest before sampling
# ([Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)). The reason it helps: the vocabulary
# has tens of thousands of tokens, and although each unlikely token has little probability on its
# own, together the long tail holds enough that you will sometimes draw from it. Those draws are
# where off-topic or garbled tokens come from, so cutting the tail keeps generation coherent
# without forcing the model to always take the single top token.
#
# This matters when you read claims about how a model behaves. `p`, temperature and the other
# sampling settings are chosen when the model is served, not fixed in the weights, and they change
# the output. The same checkpoint run with different settings can behave noticeably differently,
# and those settings usually aren't written in the model card.


# %%
@exercise
def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out everything outside the nucleus and renormalise what's left.

    Keep the smallest set of highest-probability tokens whose mass reaches `p`. The token that
    takes the running total over the threshold is inside the nucleus, not outside it. Always
    keep at least one token, whatever `p` is.

    Return a tensor of the same shape as `probs`, summing to 1, with the surviving tokens in
    the same relative proportions they started in.
    """
    ordered, index = torch.sort(probs, descending=True)
    keep = (ordered.cumsum(0) - ordered) < p
    keep[0] = True
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[index[keep]] = True
    filtered = probs * mask
    return filtered / filtered.sum()


lab.check_top_p_filter(top_p_filter)

# %%
probs = next_token_probs(logits)
for p in (0.9, 0.5):
    filtered = top_p_filter(probs, p)
    kept = int((filtered > 0).sum())
    print(f"p = {p}: filtered shape {tuple(filtered.shape)}, {kept} of {probs.shape[0]} survive, sum {float(filtered.sum()):.3f}")

# %%
# Sampling is the usual reason a "deterministic" model gives different answers across runs; even
# at temperature 0, batch composition and non-deterministic GPU kernels can still slightly nudge the output.
nucleus = top_p_filter(next_token_probs(logits, temperature=1.0), 0.9)
draws = torch.multinomial(nucleus, num_samples=8, replacement=True)
print(f"nucleus: shape {tuple(nucleus.shape)}   draws: shape {tuple(draws.shape)}")
print("eight samples from the nucleus at T=1.0:")
print("  " + "  ".join(repr(tokenizer.decode([t])) for t in draws.tolist()))
print(f"greedy (argmax) always gives: {tokenizer.decode([int(probs.argmax())])!r}")

# %% [markdown]
# *Greedy decoding* means taking the argmax (the single most likely token) at every step. It's
# repeatable: the same prompt gives the same output every time. Any temperature above 0 adds
# randomness to the choice, trading that repeatability for variety in what comes out. Hugging
# Face's [generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
# walks through the common options.

# %% [markdown]
# ## Part 2: scoring a string the model didn't write
#
# So far we've sampled from the distribution to produce text. Scoring turns that around: instead
# of asking the model to generate, you hand it a specific continuation and ask how much
# probability it assigned to exactly that string. Nothing is generated.
#
# This is more convenient than generating for most measurement tasks, because it gives you a
# number you can compare directly instead of text you have to read and interpret. A multiple-choice
# eval scores each option and takes the highest, which is what Part 3 does. Tomorrow's lab uses the
# same scorer to measure which of two replies a model prefers. You write it once here and reuse it.
#
# The mechanics: run the prompt and completion through together, and at each position read off the
# [log probability](https://en.wikipedia.org/wiki/Log_probability) the model gave the token that
# actually came next (the log of its softmax probability, so a negative number, and adding logs is
# the same as multiplying probabilities). Sum those over the completion's tokens only. Position
# *i*'s logits predict token *i+1*, so an off-by-one is the classic bug here. The check will catch
# it.
#
# The shapes the scorer moves through, before you write it:

# %%
p_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
c_ids = tokenizer(" Paris", add_special_tokens=False, return_tensors="pt").input_ids
both = torch.cat([p_ids, c_ids], dim=-1)
print(f"prompt ids       {tuple(p_ids.shape)}")
print(f"completion ids   {tuple(c_ids.shape)}")
print(f"concatenated     {tuple(both.shape)}   (batch, positions)")
with torch.no_grad():
    demo_logits = model(both.to(model.device)).logits
print(f"model logits     {tuple(demo_logits.shape)}   (batch, positions, vocab)")


# %%
@exercise
def sequence_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Total log probability of `completion` following `prompt`, in nats (natural-log units).

    Sum over the completion's tokens only. The prompt is what you condition on, not something
    you score. One forward pass over the two concatenated is enough, with no generation.
    Return a plain `float`.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([prompt_ids, completion_ids], dim=-1).to(model.device)

    with torch.no_grad():
        logits = model(ids).logits

    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    per_token = logprobs.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(per_token[-completion_ids.shape[-1] :].sum())


lab.check_sequence_logprob(sequence_logprob, model, tokenizer)

# %%
prompt = "Q: What is the capital of France?\nA:"
for answer in (" Paris", " Lyon", " Bangkok"):
    print(f"{sequence_logprob(model, tokenizer, prompt, answer):8.3f}   {answer!r}")

# %% [markdown]
# Usually what you want is the gap between two scores, since it tells you which continuation the
# model finds more likely and by how much. One thing to watch before you rely on it. Log
# probabilities are always negative, so every extra token pushes the total further down, and a
# longer string ends up with a lower score just for being longer. Compare answers of different
# lengths by their raw totals and the shortest one wins by default. Dividing each total by its
# number of tokens, giving the average log probability per token, removes that bias. That's what
# Part 3 does.

# %% [markdown]
# ## Part 3: a multiple-choice evaluator from scratch
#
# A multiple-choice benchmark is less involved than it sounds. You take a question and a few
# candidate answers, score each answer as a continuation of the question with the function you
# just wrote, and treat the highest-scoring answer as the model's pick. Do that over a set of
# questions whose correct answers you already know, count how often the model's pick is right, and
# that fraction is the accuracy. The big public benchmarks are the same procedure with more items:
# [MMLU](https://arxiv.org/abs/2009.03300) has thousands, and
# [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the standard tool
# that runs them.
#
# Score each answer with the per-token average from Part 2, not the raw total. The candidate
# answers have different lengths, and as we just saw, a raw sum would quietly favour the shortest
# one whether or not it's correct.


# %%
@exercise
def score_choices(model, tokenizer, question: str, choices: list[str]) -> list[float]:
    """Score each choice as a continuation of `question`, returning one number per choice.

    Use the mean per-token log probability of each choice, so a longer answer isn't penalised
    for its length. The evaluator picks the choice with the highest score. Build this on your
    `sequence_logprob` from Part 2.
    """
    scores = []
    for choice in choices:
        n = len(tokenizer(choice, add_special_tokens=False).input_ids)
        scores.append(sequence_logprob(model, tokenizer, question, choice) / n)
    return scores


lab.check_score_choices(score_choices, model, tokenizer)

# %%
items = lab.mc_items()
correct = 0
for item in items:
    scores = score_choices(model, tokenizer, item["q"], item["choices"])
    pick = max(range(len(scores)), key=lambda i: scores[i])
    correct += int(pick == item["answer"])
    mark = "OK " if pick == item["answer"] else "MISS"
    print(f"{mark}  {item['q'].splitlines()[0]:36}  picked {item['choices'][pick]!r}")
print(f"\naccuracy: {correct}/{len(items)}")

# %% [markdown]
# ## What you built
#
# You now have three ways to look at a model's output distribution. Softmax with temperature turns
# logits into probabilities and sets how peaked they are. Entropy measures how spread out they are.
# Scoring reads off the probability of a specific string. The Part 3 evaluator is the scorer
# run in a loop and checked against known answers, which is the core of what a benchmark number
# measures.
#
# Two things worth keeping. First, when a system card reports something like 74% on a benchmark,
# you know the operation behind that number and some of the ways it can be shaped: how the answer
# choices were worded, whether length bias was corrected, which tokens were counted. Second, the
# output you get depends on decoding settings (temperature, top-p, greedy versus sampling) that are
# chosen at serving time and sit outside the weights. The next lab picks up that thread: once you
# see that the settings and the prompt around a model change what it does, a good part of a model's
# apparent "behaviour" turns out to belong to whoever assembles those, not to the weights alone.

# %%
# @lab-only
# Stuck on sequence_logprob? Print the shapes. logits[0] has one row per input position, and
# row i scores the token at position i+1. So the rows you want end at the last position, and
# the target tokens start one index later.
print("row i predicts token i+1")
