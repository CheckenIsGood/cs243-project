# Theory: optimal result consumption under a completion deadline

Status: derived 2026-11-06; independently reviewed the same day and re-reviewed in a third pass (proofs
re-checked; Props 0–4 survived random searches of >10,000 instances each; false claims and gaps fixed — see
Section 9). Propositions 0–4 have numerical checks in `tests/test_theory.py`; the Section 8 extensions
(Props A1–A3, B1–B5) in `tests/test_extensions.py`. Section 6 and 8 numbers come from
`scripts/explore_regimes.py`, `scripts/explore_speculation.py` and `scripts/explore_preload.py` and are
**model predictions from guessed costs**, not measurements. Nothing in Section 8 has been run on vLLM yet.

## 1. Restricted problem (offline, deterministic)

One task, n admitted results N with release times r_i, one nonpreemptive model slot, idle at time 0.
A result may be summarized (in a batch B, cost c(A,B) after set A was summarized) or left raw and read
by the final invocation (cost f(S) with summaries S and raw N\S). Every result must be consumed.
Costs are known, nonnegative and time-independent; summary state depends only on the set summarized.
`inf` marks context-infeasible actions. R(B) = max_{i in B} r_i.

**DP (general costs).**
F(∅)=0, F(S) = min_{∅≠B⊆S} max(F(S\B), R(B)) + c(S\B, B);
C* = min_S max(F(S), R(N\S)) + f(S). Time O(3^n + n2^n), space O(2^n). (`restricted.solve_dp`)

**Proposition 0 (exactness).** C* is the minimum completion time over all ordered sequences of disjoint
batches followed by one final invocation.
*Proof.* Any schedule summarizing S ends with a last batch B after A=S\B; it cannot start before
the previous work ends or B's last release, so its end is ≥ max(F(A),R(B))+c(A,B) by induction, and the
DP predecessor schedule attains it. Keeping only the earliest F per set is valid because costs depend on
sets only and an earlier state can idle to emulate a later one. Finalization similarly starts at
max(F(S), R(N\S)). ∎ Verified against independent forward enumeration (`solve_exhaustive`), 150 random
instances with A-dependent costs and infeasible masks.

## 2. The cost model we actually expect

Each vLLM invocation: latency = α + β_p·(prompt tokens) + β_d·(output tokens). Raw results have L tokens,
summaries s < L tokens; an evidence update for b results emits o_0 + o·b tokens; the final answer emits o_f.
So c depends only on (|A|, |B|), f only on |S|, and the context cap C_max makes some counts infeasible.
Define δ = β_p (L − s): the final-invocation time saved per summary. Note c(·,b) ≥ β_p·L·b ≥ δ·b
(strictly greater whenever s > 0 or α, base prompt or decode cost is positive). Release times are r_i ≥ 0.
(`affine.AffineCostModel`)

Propositions 1–3 assume this Section 2 cost model: f(S) = f(∅) − |S|·δ and c(A,B) ≥ δ·|B|. (Props 1–2 use
only these two facts, with f(S) ≥ f(∅) − |S|·δ sufficing, so they also hold for set-dependent summary costs;
Prop. 3 needs costs that depend on counts only. See Prop. A2.)

**Proposition 1 (simultaneous returns ⇒ fold).** If all results return at the same time t₀ and the all-raw
final prompt fits, folding everything into the final invocation is optimal.
*Proof.* Every batch starts at or after t₀, so a schedule that summarizes S starts its final at
≥ t₀ + Σ_B c(A,B) ≥ t₀ + |S|·δ and finishes at ≥ t₀ + |S|·δ + f(∅) − |S|·δ = t₀ + f(∅) = C_fold. ∎

**Proposition 2 (fold is near-optimal when context does not bind).** If n ≥ 1 and the all-raw prompt
fits, C_fold − C* ≤ (n−1)·δ, for any release times r_i ≥ 0. (The bound is tight; see below.)
*Proof.* C_fold = r_max + f(∅). Let j be a last-released result. If j is raw in an optimal schedule, the
final starts ≥ r_max and f(S) ≥ f(N\{j}) = f(∅) − (n−1)δ. If j is summarized in batch B (after set A), the
final starts after that batch ends, ≥ r_max + c(A,B) ≥ r_max + δ, and f(S) ≥ f(N) = f(∅) − nδ, so again
C ≥ r_max + f(∅) − (n−1)δ. ∎ *Tightness:* release n−1 results at 0 and one at T ≥ (n−1)·max c; the optimal
schedule summarizes the n−1 early results before T, finishing at T + f(∅) − (n−1)δ.

*Consequence for experiment design:* with ~768-token results and n ≤ 5 (Section 6 costs:
δ = 0.235 s, so (n−1)δ ≤ 0.94 s), the time gap is under a second (prefill is fast; summaries cost decode). Prop. 2 bounds the *time* gap only: near a deadline even
this sub-second gap can change the miss rate (Section 6, context-free row), but the effect is small and
confined to deadlines at the edge of the completion-time distribution. Large gains require **(a)** a binding
context cap, or come from **(b)** answer-quality effects of summaries. So the workload makes the context bind
(larger results / tighter cap) — a theory-driven design choice, stated as such.

**Proposition 3 (prefix structure ⇒ O(n²) exact solver).** With count-based costs, some optimal schedule
summarizes a prefix of the results in release order, in batches contiguous in that order.
*Proof.* Take an optimal schedule with batch sizes b_1..b_m (k = Σb). Reassign, in release order, the first
b_1 results to batch 1, the next b_2 to batch 2, etc. Costs are unchanged (they depend on counts only).
The new batch t waits for the (b_1+…+b_t)-th order statistic, which is ≤ max release among the original
batches 1..t, and the original batch t could not start before that maximum; by induction every new batch
ends no later. The final waits for r_max if any result stays raw; originally either the max-release result
was raw (same wait) or it was summarized (so the last batch ended after r_max). ∎
DP: F[k] = min_{j<k} max(F[j], r_(k)) + c(j, k−j); C* = min_k max(F[k], r_(n)·[k<n]) + f(k).
(`affine.prefix_dp`; equals the subset DP on 150 random instances.)

## 3. Why the offline optimum is not a policy

The DP needs future release times. Plugging in predictions gives a certainty-equivalent plan, which never
hedges (e.g. it will start a 4 s summary just before an unexpectedly early last return). We therefore
solve the online problem exactly where we can.

## 4. Exact optimal online policy (Experiment A setting)

Assumptions: the n calls are launched together (K = n, Experiment A); admission-to-return delays are
i.i.d. from a known discrete distribution (our controlled delay regimes are exactly this; the calibrated
empirical distribution is the estimate); count-based costs; decisions when the model is idle.

**Proposition 4 (sufficient statistic).** Under these assumptions the state (t, r = #returned,
k = #summarized) is sufficient: identities of returned results do not matter (count-based costs), and
every pending call has the same conditional residual distribution P(T ≤ t' | T > t) because all share age t.
New returns during an action of length τ are Binomial(n−r, (G(t+τ)−G(t))/(1−G(t))).

Actions: CONSUME(b), 1 ≤ b ≤ r−k (context-feasible); WAIT one step (r<n); FINAL (r=n and raw prompt fits).
For any nondecreasing loss ℓ(C) — miss indicator, tardiness, or completion time — backward induction
V(t,r,k) = min_a E[V(next)] with V = ℓ(C) at FINAL gives the **optimal nonanticipative policy** for this
model. Precisely: optimal among policies that decide on the time grid dt, with consume durations rounded up
to whole steps; delays must have bounded support (true for our controlled regimes) so the recursion is finite. States with r = n are deterministic: V(t,n,k) = ℓ(t + g(k)) where g is the remaining-time DP.
State space ≈ (#time steps) × (n+1)(n+2)/2, solved in < 0.1 s for n=4, dt=0.1 s; the policy is a lookup table, so
runtime controller cost is negligible. (`online_mdp.ConsumptionMDP`; checks: dominates every static
policy under three objectives, exact evaluation matches 4000-run Monte Carlo, reduces to the offline DP
for point-mass delays — i.e. simultaneous releases, the only deterministic case i.i.d. delays allow —
never consumes an unreturned result.)

This is optimal only for the stated model; the real system has cost noise, variable result lengths and
retrieval jitter. Report it as "optimal under the calibrated model", and measure the transfer.

## 5. Heuristic for when the assumptions break: required-summary slack (RSS)

Inspecting the optimal table (n=4, context needs m=2 summaries, miss objective with a small tie-break
toward earlier completion, tie_eps = 1e-3) shows a simple structure:
1. It summarizes exactly m = min #summaries for the final prompt to fit. This is an empirical observation
   about this table, not a consequence of Prop. 2: an extra summary done in idle time is off the critical
   path, and can save up to δ. With no tie-break (tie_eps = 0) many plans tie on miss probability and the
   table often ends with 3–4 summaries; with the context free (m = 0) the optimal policy frequently makes
   1–3 summaries in idle time — that is precisely the small Prop. 2 gap.
2. It summarizes those m while the remaining calls are still out — using idle model time.
3. With fewer than m ready, it waits for another result only if one is likely to arrive within the time a
   combined batch saves (one batch of 2 is cheaper than two singles: shared overhead and prompt).
4. After m summaries, it waits for everything and folds the rest raw.

`online_mdp.required_summary_slack` implements this rule with the survivor distribution and current token
counts. Under the model it is within 4 points of the optimal miss probability in the context-binding
regimes tried (D = 16–24 s). When the context does not bind, m = 0 and RSS reduces to fold, which can be up
to 7 points worse than optimal (D = 18 s, variable delays), because RSS ignores the small Prop. 2 time gap
that the optimal policy exploits.
Generalizations for later experiments (staggered launches K<n, heterogeneous result lengths, tool types):
compute m from actual token counts; summarize the *longest* ready results first (most context relief per
decode token); use per-call survivor residuals in step 3; or use the certainty-equivalent prefix DP with
survivor-quantile releases as a replanning fallback.

## 6. Model-predicted separation (guessed costs — replace with Gate 1 profiles)

α=0.08 s, prefill 3000 tok/s, decode 30 tok/s, n=4, final 192 output tokens, evidence update 32+64b tokens.
Variable delays: U[0.5,4] s w.p. 0.7, U[4,12] s w.p. 0.3. Predicted P(miss) (`scripts/explore_regimes.py`,
including its "fine deadlines" block for the predictable rows):

| Setting | D | optimal | RSS | best static (fold) | incremental |
|---|---|---|---|---|---|
| context binds (L=1500, cap 4096) | 18 s | 0.27 | 0.30 | 0.83 | 1.00 |
| context binds | 20 s | 0.01 | 0.01 | 0.66 | 1.00 |
| context free (L=768, cap 8192) | 18 s | 0.16 | 0.23 | 0.23 | 1.00 |
| predictable U[2,3] s, context free | 9–10.75 s (sweep) | = fold | = fold | — | worse |
| predictable U[2,3] s, context binds | 16.5 s | 0.70 | 0.70 | 0.99 | 1.00 |
| predictable, context binds | 16.85 s | 0.14 | 0.14 | 0.84 | 1.00 |
| predictable, context binds | 17.0 s | 0.065 | 0.065 | 0.72 | 1.00 |

(Predictable rows use deadlines between integers: at integer D the miss probability is 0 or 1 for every
policy because fold's completion lies in [16.22, 17.22] s. In the predictable, context-free sweep the largest
fold − optimal gap is 0.000.) "Best static" is fold in every row shown; it is not fold everywhere (context
binds, variable delays, D = 24 s: batch-of-2 0.22 < fold 0.30).

Reading: incremental processing is hopeless when decode is slow (Section 8.1 shows this changes once summary
decode is 2–3× faster). Fold matched the optimum at every deadline tried only in the predictable, context-free
setting. With a free context it is within (n−1)δ in time but can still lose ~7 points of miss rate near the
deadline. When the context binds, the adaptive policies win under both delay regimes — with predictable
delays the gain is concentrated in a narrow deadline band; with variable delays it is spread over a wider
band. Whether real costs land in these regimes is the Gate 1 regime check.
Model parameters for this section: α = 0.08 s, base prompts 300 tokens, s = 64, o_0 = 32, o = 64, o_f = 192.

## 7. Claim boundaries

Propositions 0–3: exact for the deterministic restricted model only. Proposition 4: optimal online policy
only for i.i.d. known bounded delays, simultaneous launch, deterministic count-based costs, and decisions on
the dt grid. RSS: heuristic.
Section 8.1 (speculative decoding): Props 0–4 hold verbatim for fixed per-type decode rates; Props 1–2 hold
exactly for content-dependent summary speedups (Prop. A2); `StochasticDurationMDP` is exact only when action
durations are independent of each other and of the tool delays, with laws depending on (action, k, b) only.
Section 8.2 (pre-loading): Props B1–B4 are exact for the deterministic pre-load model (evidence-log layout, no
eviction, prefill linear in uncached tokens, canonical summaries, consume prompts sharing no prefix with the
final); B2 also holds online (argument given); B3 online is a numerical observation; `PreloadMDP` is optimal
only within removal-free policies. Section 8 numbers are model predictions from guessed costs (speedups,
preload overhead α' = α), not measurements; nothing about vLLM's actual speculation acceptance rates or cache
behavior has been measured.
None of this optimizes answer quality; quality differences between summarized and raw evidence are measured
separately. The DP/MDP are standard techniques applied to our model, not new scheduling theory.

## 8. Engine-dependent extensions (implemented and tested in the model; not yet run on vLLM)

The cost model is parameterized by the inference engine. Two vLLM features change it in different ways.
Code: `affine.AffineCostModel` (new optional fields), `speculation.py`, `cache_model.py`, `preload_mdp.py`;
checks: `tests/test_extensions.py`; predictions: `scripts/explore_speculation.py`, `scripts/explore_preload.py`.

### 8.1 Extension A: prompt-lookup speculative decoding

**Model.** Speculation lowers the *effective* decode cost per token, more for summaries (they copy from the
prompt) than for the final answer. `AffineCostModel.consume_decode_s_per_tok` = β_d^sum and
`final_decode_s_per_tok` = β_d^fin, both defaulting to `decode_s_per_tok` (so every earlier result is
bit-for-bit unchanged; tested). Prefill is not sped up: the draft tokens are verified by the target model,
so a summary of b results still prefills b·L raw tokens. With greedy decoding, speculation is intended to be
output-preserving; verify equal outputs, do not assume.

**Proposition A1 (fixed per-type rates).** If β_d^sum and β_d^fin are constants, costs remain deterministic and
count-based with c(k,b) ≥ α + β_p(base + L·b) ≥ δ·b and f(S) = f(∅) − |S|·δ (δ is unchanged: it involves
prefill only). Hence Props 0–3 hold verbatim, and Prop 4 holds verbatim (the MDP uses costs only through
c and f). *Checks:* prefix DP = subset DP and Props 1–2 on 120 random instances with random per-type rates.

**Proposition A2 (content-dependent or random summary durations).** Props 1–2 use only (i) every summary of
b results takes ≥ β_p·L·b ≥ δ·b, and (ii) f(S) ≥ f(∅) − |S|·δ. (i) holds for *any* decode speedup.
(a) Deterministic but set-dependent summary costs c(A,B) (speedup depends on which results): Props 0, 1, 2 are
exact (the proofs go through unchanged); Prop 3 is not (costs are no longer count-based).
(b) Random durations: if the final's duration is its deterministic prefill part plus a decode time whose law
does not depend on S and is independent of the past, then with simultaneous returns any (even adaptive)
policy's completion stochastically dominates fold's (C ≥ t₀ + |S|δ + F₀ − |S|δ = t₀ + F₀ with F₀ distributed
as fold's final), so fold minimizes E ℓ(C) for every nondecreasing ℓ. Prop 2 holds pathwise when fold and
the comparison schedule are coupled through the same final decode time. *Check:* 60 random instances with
per-(A,B) speedups in [1,4].
So "fold is near-optimal when the context fits" survives every decoding speedup. (The earlier wording "only
Prop 0 is exact; Props 1–3 hold for the fitted mean costs" understated this and was misleading; see Section 9.)

**Proposition A3 (online, random durations).** Under Prop 4's assumptions, let each CONSUME(k,b) and FINAL(k)
duration be drawn independently of the tool delays and of every other duration, from a law depending only on
(action, k, b). Then (t, r, k) is still sufficient: new returns during an action of random length τ are
Binomial(n−r, (G(t+τ)−G(t))/(1−G(t))) given τ, and averaging over τ's law is exact
(`speculation.StochasticDurationMDP`). If durations depend on *which* results are summarized, identities
matter and the model is an approximation. *Checks:* with no speedup law it equals `ConsumptionMDP`; a
point-mass speedup equals the deterministic per-type-rate MDP; the optimum dominates all static policies, RSS
and the deterministic-mean policy; exact evaluation matches 3,000-run Monte Carlo.

**Deterministic-mean approximation.** Replacing random speedups by β_d·E[1/factor] is not safe: for the
speedup law {1×: 0.25, 2×: 0.35, 3.5×: 0.40} (context binds, variable delays) the mean model mispredicts the
optimal miss probability by up to 0.06 (D = 16 s: 0.42 vs exact 0.48), and its policy, run in the random
model, loses up to 9 points to the exact optimum (D = 18 s: 0.30 vs 0.21; RSS: 0.24).

**Model predictions** (`scripts/explore_speculation.py`; guessed costs of Section 6, assumed speedups, not
measured). Variable delays, P(miss); "P(k>m)" is the probability that the optimal policy (tie_eps = 1e-3, as in
Section 5) makes more than the m required summaries.

| context binds, D | speedup (sum/final) | optimal | RSS | fold | batch2 | incremental | P(k>m) |
|---|---|---|---|---|---|---|---|
| 18 s | 1×/1× | 0.266 | 0.303 | 0.830 | 1.000 | 1.000 | 0.00 |
| 18 s | 2×/1× | 0.184 | 0.232 | 0.605 | 0.557 | 0.495 | 0.53 |
| 18 s | 3×/1× | 0.179 | 0.230 | 0.535 | 0.450 | 0.346 | 0.62 |
| 16 s | 3×/1.5× | 0.164 | 0.206 | 0.518 | 0.440 | 0.335 | 0.62 |
| context free, 18 s | 1×/1× | 0.157 | 0.229 | 0.229 | 0.986 | 1.000 | 0.98 |
| context free, 18 s | 3×/1× | 0.134 | 0.229 | 0.229 | 0.393 | 0.319 | 1.00 |

Reading: (i) faster summaries move the whole curve to earlier deadlines; (ii) the best static policy changes
from fold to incremental at 2–3× in the binding regime (D ≥ 16–18 s), so "incremental is hopeless" is a
slow-decode statement; (iii) RSS's gap to the optimum grows modestly — at most 3.7 points (1×), 4.8 (2×), 5.1
(3×) when the context binds, and 7.2 → 9.5 points when it is free, because RSS never makes extra summaries;
(iv) the Section 8 testable prediction holds in the model: with the context binding, extra (> m) summaries go
from never (1×) to probability 0.37–0.53 (2×) and 0.60–0.62 (3×) across D = 10–20 s.

### 8.2 Extension B: KV pre-loading through automatic prefix caching

**Prompt layout (recommended): the evidence log.**
final prompt = P0 · W · X · tail, where
- P0 = system prompt, answer instructions/schema and the question — everything that is known at time 0;
- W = evidence blocks in the order they *entered* the working context: a raw result enters when a PRELOAD
  appends it; a summary enters, appended at the end, when its CONSUME finishes;
- X = returned raw results not yet in W, appended by the final invocation;
- tail = chat-template end of turn + assistant header (a few tokens, never cached).
Justification: prefix caching reuses only an unchanged token prefix, so (1) stable content goes first;
(2) every insertion is an append, so caching raw results never has to be redone; (3) a summary is appended
after the cached raws rather than inserted before them, so summarizing a result that was never pre-loaded
invalidates nothing; (4) the only invalidating edit is deleting a pre-loaded raw block (when that result is
summarized), which invalidates every cached token after it — and by Prop. B1 optimal plans never need it. The
current runtime layout (summaries sorted by call id, *then* raws) inserts each new summary before all cached
raws and invalidates them; in the exhaustive model the evidence log was never worse on 1,000 random instances
and is strictly better on the B4 instance below (14.24 vs 14.74 s).

**Cost model and assumptions** (`cache_model.PreloadCostModel`). PRELOAD sends P0·W' (W' = W plus q appended
raws) with max_tokens = 1 and costs α' + β_p·(uncached tokens of P0·W'), no decode. CONSUME uses its own
prompt (different system prompt; shares no cached prefix with P0·W) and costs the base c(k,b). FINAL costs
α + β_p·(uncached prompt tokens) + β_d^fin·o_f. Context: every prompt must fit the cap. State
(k, p, j, z): k summaries, p raws in W, j summaries appended since the last PRELOAD (uncached), z = P0 cached;
then FINAL's uncached tokens are P0·[z=0] + s·j + L·(n−k−p) + tail. Assumptions: no eviction (one sequence at
a time, task-sized KV pool); block granularity ignored (vLLM caches only full blocks, ≤ block_size−1 tokens per
request); prefill time linear in uncached tokens (attention of new tokens over a long cached context ignored);
the 1 generated token is folded into α'; summaries are canonical (a result's summary text does not depend on
its batch); preload and final token IDs share the exact prefix. vLLM also caches the consume instruction
prefix across consumes; that only lowers c(k,b) for k > 0, stays count-based, and can be folded into c.

**Proposition B1 (exact solver).** (a) *Removal-free:* some optimal schedule never summarizes a pre-loaded raw.
Proof: drop that raw from the PRELOAD that appended it (or drop the PRELOAD if it appended nothing else). Every
action is no longer, starts no later (earliest-start timing is monotone), and every later prompt's cached
prefix is at least as long (before the deletion the original's prompts contain the doomed block; after it, the
two W sequences coincide and the modified cache never lost the blocks behind it). (b) *Release-order use:* in a
removal-free schedule all costs depend only on (k, p, j, z) and the action counts, and each action's release
constraint is the latest release among the results it uses; reassigning identities so that results are used
(summarized *or* appended raw) in release order along the action sequence never delays any action — the Prop. 3
argument with "use" in place of "summarize". (c) Hence a DP over (k, p, j, z) with earliest completion times,
O(n³) states and O(n) transitions, is exact (`solve_preload_dp`). *Checks:* equals an independent brute force
over explicit identities, all subsets and preload orders, deletions allowed, with an explicit prefix-trie cache
(`solve_preload_exhaustive`) on 3,000 random instances with n ≤ 4 (half with adversarial costs) and 160 in the
test suite; with PRELOAD disabled it equals `prefix_dp`.

**Proposition B2 (replaces Prop. 2).** Let Δ_B = max over feasible (k,b) of [α' + β_p·(P0 + L·b) − c(k,b)]⁺ — how
much longer pre-loading b raws together with the whole final prefix can take than summarizing them. If the
all-raw final prompt fits, then C(fold + preload) − C*(preload) ≤ (#summary batches of an optimal schedule)·Δ_B ≤
n·Δ_B. In particular **if Δ_B = 0, fold with pre-loading is exactly optimal**: when the context does not bind,
summarizing saves no time at all. Proof: replace each CONSUME(B) of an optimal schedule by a PRELOAD appending
B's raws at the same position (same release constraint, at most Δ_B longer; the original PRELOADs only get
shorter because no summaries need caching); the final reads the same raws uncached as before and no uncached
summaries; delays add up over at most n replacements. Δ_B = 0 is implied by (P) below and holds for the
Section 6 costs, also with 3× faster summaries. If Δ_B > 0 the gap can be positive (e.g. α' = 0.8 s with very
fast decode: 0.70 s gap ≈ 1.5·Δ_B). Online: the same replacement, idling until the original action would have
ended, is a nonanticipative policy that finishes no later (rounded preload duration ≤ rounded consume
duration), so under Prop 4's assumptions some optimal pre-load policy never summarizes. *Checks:* 13,413
random instances (bound never violated; gap 0 whenever Δ_B = 0); online: `PreloadMDP` capped at 0 summaries
equals the full optimum in the context-free settings. Consequence for Prop 1: with simultaneous returns at
t₀ > 0 the slot *is* idle before t₀, so plain fold is no longer optimal; fold plus pre-loading P0 is (B2).

**Proposition B3 (exactly m summaries).** Under (P): α' + β_p·P0 ≤ β_d^sum·o (pre-loading one raw plus the final
prefix is no slower than *decoding* one result's summary), some optimal pre-load schedule summarizes exactly
m = min_summaries_needed results. Proof: an extra summary inside a batch of b can be split off into a PRELOAD of
its raw: c(k,b') + α' + β_p(P0 + L(b−b')) ≤ c(k,b) by (P) and the affine form of c; caching pending summaries
earlier in that PRELOAD shortens a later action by the same amount; later consumes see smaller k (c is
nondecreasing in k); the final with m summaries fits by definition of m. Section 6 costs: 0.175 s ≤ 2.13 s
(0.71 s at 3× speedup). *Checks:* 9,732 random instances satisfying (P). Online (not proven, because rounding
two split actions up can exceed one): capping `PreloadMDP` at m summaries lost nothing in every Section 6
setting tried (script column "opt+pre,k<=m", tests). So with pre-loading, RSS rule 1 ("exactly m") stops being
merely empirical: the extra idle-time summaries the plain optimum uses are replaced by pre-loads.

**Proposition B4 (the summarization order can flip).** Prop. 3's earliest-first structure fails once pre-loading
exists. Instance (Section 6 costs, L = 1500, cap 4096 ⇒ m = 2, α' = α, P0 = 284 cacheable + 16 tail tokens):
r = (0, 0.7, 0.7, 0.7) s. The optimum pre-loads P0 + raw₀ in [0, 0.675], summarizes {1, 2} as one batch in
[0.7, 7.213] and finalizes at 14.241 s. The best schedule that summarizes the earliest results {0, 1} finishes at
14.741 s: result 0 cannot be summarized before 0.7 s without giving up the cheaper batch, and an idle slot can
pre-load a result only if it stays raw. Without pre-loading both orders tie (14.836 s, Prop. 3). Confirmed by the
brute force, including the earliest-first restriction (test `test_order_flip_counterexample`). The mechanism is
not cache invalidation (B1 plans never invalidate) but use of idle time before a batch's last release. What
survives of Prop. 3 is B1(b): results are *used* in release order; which uses are summaries is decided by the DP.
`earliest_first_optimum` computes the best earliest-first schedule for comparison.

**Proposition B5 (online optimum with pre-loading).** Under Prop 4's assumptions plus this cost model and
layout, restricted to removal-free policies, the state (t, r, k, p, j, z) is sufficient and backward induction
is exact (`preload_mdp.PreloadMDP`; actions CONSUME(b ≤ r−k−p), PRELOAD(q ≤ r−k−p; q = 0 caches P0/pending
summaries), WAIT, FINAL). Removal-freeness is without loss offline (B1a) but is an assumption online.
*Checks:* a never-pre-loading policy evaluates exactly as in `ConsumptionMDP`; the optimum dominates the plain
optimum and every heuristic with idle pre-loading under three objectives; point-mass delays = offline DP with
the same step-rounded costs; exact evaluation matches 3,000-run Monte Carlo. ≈ 0.05 s per solve (n = 4, dt = 0.1 s).

**Model predictions** (`scripts/explore_preload.py`; guessed costs, α' = α = 0.08 s; "+pre" = the same policy
pre-loading P0, pending summaries and the returned raws it will not summarize whenever it would wait).

| setting, D | optimal → +pre | RSS → +pre | fold → +pre |
|---|---|---|---|
| context binds, variable, 18 s | 0.266 → 0.186 | 0.303 → 0.224 | 0.830 → 0.752 |
| context binds, variable, 16 s | 0.796 → 0.744 | 0.814 → 0.778 | 0.996 → 0.988 |
| context binds, predictable, 16.5 s | 0.699 → 0.290 | 0.699 → 0.539 | 0.994 → 0.946 |
| context binds, predictable, 16.85 s | 0.140 → 0.013 | 0.140 → 0.065 | 0.836 → 0.439 |
| context free, variable, 18 s | 0.157 → 0.114 | 0.229 → 0.114 | 0.229 → 0.114 |
| context free, predictable, 10.0 s | 0.983 → 0.258 | 0.983 → 0.324 | 0.983 → 0.324 |
| binds + 3× summary decode, variable, 18 s | 0.179 → 0.145 | 0.230 → 0.145 | 0.535 → 0.485 |

Expected completion (variable delays): context binds, optimal 17.30 → 17.06 s, fold 21.63 → 21.09 s; context
free, optimal 15.00 → 14.18 s, fold 15.01 → 14.18 s. Incremental gains ≤ 0.1 s (it has almost nothing raw).
Reading: pre-loading removes at most the prefill of what returns early (≈ 0.5 s per 1,500-token result,
≈ 0.1 s for P0), so it moves miss rates a lot only where the completion-time distribution is steep around the
deadline (predictable delays: 0.14 → 0.013 for the optimum), and by up to 8 points under variable delays. With a
free context, fold + pre-loading equals RSS + pre-loading and is within 0.001 of the pre-load optimum under
variable delays, as B2 predicts — RSS's 7-point deficit disappears. Greedy idle pre-loading is not always
enough: in the predictable band it is up to 25 points (binding, 16.5 s) or 7 points (free, 10.0 s) worse than the
pre-load optimum, which times and batches its pre-loads (in the free case the best never-summarizing pre-load
policy, column "opt+pre,k<=m" with m = 0, attains the optimum, so the loss is pre-load timing, not summarization).

**What the runtime needs to implement B** (for Stage B; verify every flag against the pinned vLLM version):
1. Prefix caching on, and per-request cached-token counts in the usage report (vLLM can report
   `prompt_tokens_details.cached_tokens`; check the flag), so the model's predicted uncached tokens can be
   checked per request. Reset the prefix cache between independent tasks (PLAN fairness rule).
2. Final prompt in evidence-log order: P0 (system + answer schema + question + evidence header), then blocks in
   entry order — raw blocks when pre-loaded, summary blocks appended when their consume returns — then the
   still-unlogged raws, then the template tail. No sorting of summaries by id; no summaries-before-raws section.
3. Exact token-prefix property: build token IDs by concatenating per-block tokenizations (fixed separators),
   or render the full chat-templated prompt and assert that the pre-load's IDs are a prefix of the final's;
   send both as token IDs to the completions endpoint; PRELOAD uses max_tokens = 1 and its output is discarded.
4. PRELOAD is a slot action: it occupies the single model slot and its time counts toward the deadline.
5. Removal-free rule: never summarize a result whose raw block was pre-loaded (the policy guarantees this);
   if it ever must, everything after that block is uncached.
6. Controller state per task: the block list W and (k, p, j, z); policy = `PreloadMDP.optimal_policy()` lookup
   (t, r, k, p, j, z), or `with_idle_preload(mdp, required_summary_slack(mdp))` as the heuristic.
7. Gate 1 calibration: α' (overhead of a 1-token request), prefill time vs uncached tokens at long cached
   context, and whether summaries are canonical enough for the cached-token accounting to hold.

## 9. Review log

2026-11-06, second pass (author, line by line): Props 0–4 proofs re-derived and correct as stated.
Clarified that Props 1–3 depend on the Section 2 cost model, wrote Prop 1's inequality chain explicitly,
replaced the undefined c_min in Prop 2 with c(A,B) ≥ δ, and added an explicit tightness example.

2026-11-06, independent review (scripts in the session scratchpad: offline_checks.py, mdp_checks.py,
policy_structure.py, predictable_fine.py). Found and fixed: (1) the predictable-delay row of Section 6 was
false when the context binds (fold loses up to 70 points at deadlines between integers); (2) "fold is optimal
with predictable delays or a free context" was false; (3) Section 8.1 overstated which propositions survive
content-dependent speculation; plus gaps in the RSS rule-1 justification, Section 8.2 claims, the "(n−1)δ
under a second" scope, and the status line. Verified: Props 0–3 proofs and random searches (DP = exhaustive on
3,000 general instances; prefix_dp = subset DP on 20,000; Prop 1–2 on 12,698, Prop 2 bound tight); MDP on
1,500 random instances (optimal ≤ every static policy; evaluate(optimal) = solve(); point-mass delays = offline
DP); every Section 6 number reproduces; one MDP solve ≈ 6 ms.

2026-11-06, third pass (skeptical re-review + Extensions A and B; checks in `tests/test_extensions.py`, larger
random searches in the session scratchpad: bigcheck.py, flip.py, flip2.py, flip3.py, b2gap.py). Props 0–4
proofs re-derived once more; no error found in their statements or proofs. Found and fixed:
(1) traceability: the predictable-delay rows of Section 6 (16.5/16.85/17.0 s, and "context free = fold") were
produced by a scratchpad script, not a repo script — now printed by the "fine deadlines" block of
`scripts/explore_regimes.py` (all values reproduce; 17.0 s is 0.065, previously rounded to 0.07); the vague
"0 or = fold" row is replaced by the sweep result (max gap 0.000 over 9–10.75 s); noted that "best static" is
not fold everywhere (D = 24 s, binding: batch-of-2 is better). (2) Section 6 "Fold is optimal only when …"
softened to what was tested. (3) Old 8.1 understated robustness: Props 1–2 hold *exactly* for content-dependent
summary speedups (they use only c ≥ β_p·L·b), not just "for fitted mean costs" (Prop. A2); and the claimed
exact fix for Prop 4 needed an independence condition (Prop. A3). (4) Old 8.2 "Prop 1 still holds: no idle
time" was wrong: the slot is idle before t₀, plain fold is beaten by fold + pre-loading P0 (e.g. r = (2,2,2,2):
16.136 → 16.041 s); the correct statement is B2. (5) Old 8.2 attributed the possible order flip to cache
invalidation; under the evidence-log layout optimal plans never invalidate (B1a) and the flip, which does exist
(B4, also with Section 6 costs: 24 of ~3,000 random release patterns for L = 1500/cap 4096, none with a free
context), comes from using idle time. (6) Old 8.2 left Prop 2's replacement open: B2 gives gap ≤ n·Δ_B, and 0
under Section 6 costs. New: Extension A (per-type decode fields, `StochasticDurationMDP`, predictions) and
Extension B (layout, `cache_model.py` exact DP + brute force, `preload_mdp.PreloadMDP`, predictions).
Verified: preload DP = brute force on 3,000 random instances (n ≤ 4); earliest-first DP = restricted brute force
on 747; evidence log ≤ summaries-first on 1,000; B2 on 13,413 (bound never violated); B3 on 9,732 under (P).
