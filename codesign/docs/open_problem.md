# The open problem: online deadline scheduling of agent DAGs with tool waits

Status: written 2026-11-07. Companion code: `src/deadline_agent/open_problem.py` (exact `Fraction` simulators,
exact offline optimum, all constructions). Machine checks: `tests/test_open_problem.py` (4666 tests, all
passing, about 13 s). Builds on `docs/general_theory.md` (Theorem 1, Lemmas 2 and 3), which this document does not modify.

Labels. **[proved]**: full written proof here, plus a machine check (Z3 for the algebra, exact simulation and
exact offline optimum for constructions, randomized property tests). **[known; cite]**: the source was opened and
says it (theorem number given). **[known; secondary]**: confirmed only through a secondary source or abstract.
**[known; unverified]**: not checked. **[conjecture]**: evidence given. **[refuted]**: explicit exact
counterexample.

---------------------------------------------------------------------------------------------------------------

## 0. Bottom line

**Proved.**
1. *No speed suffices without tool clairvoyance* (Thm 2). Suppose a tool's duration is revealed no earlier
   than when the tool is issued. Then no deterministic online algorithm, at any speed s, can meet all deadlines
   on every instance a speed-1 offline schedule can meet. This holds even on one machine, with all tasks
   released at 0, chains GPU→TOOL, and known GPU work and known tool upper bounds. With slack δ (every window
   ≥ (1+δ)·critical path), speed **s ≥ 1 + 1/δ is necessary**. A randomized and known-distribution version
   (Cor 2′) also holds. For on-time value, the ratio is unbounded at every speed (Thm 3): online gets 0, offline
   gets ⌊k/s − 1/2⌋.
2. *That bound is tight* (Thm 4). Global EDF by task deadline at speed **s = 1 + 1/δ** meets every deadline
   whenever a speed-1 m-machine schedule can. This holds for DAG tasks and adversarial tool durations, with no
   knowledge of tool durations or GPU work. It requires all tasks released together, or more generally
   "r_i < r_j ⇒ d_i > d_j". So for this class the speed–slack trade-off is exactly s* = 1 + 1/δ.
3. Restricted positive results:
   - Tasks with a single GPU node reduce exactly to classical deadline scheduling, provided the tool durations
     downstream of the GPU node are known (Prop 5, Cor 5′). The classical feasibility and throughput results,
     including Eberle's O(1/δ), then apply. Thm 2 shows that knowing those downstream durations is necessary.
   - Lifting Thm 4 to the batched engine through Lemma 2 costs a factor 2 in speed (Thm 6). The lower bound also
     lifts.
   - Evicting KV whenever a task waits on a tool costs exactly a (1+β) speed factor, where β bounds recompute
     work relative to node work (Thm 7).
   - Two simple negative composition facts (Props 8 and 9): keeping KV under EDF can need k times the memory,
     and a recompute-cost-optimal cache policy misses deadlines at every speed.

**Refuted.**
- R4: "EDF at speed 1+1/δ suffices for arbitrary release times". With δ = 1, EDF misses at speeds 2, 9/4 and
  5/2 on exact instances that are feasible at speed 1 and where every task has slack 1.
- R5: "critical-path EDF (CP-EDF) is speed-optimal when tool durations are known". It misses at speed 29/20 on
  a two-task, common-release, zero-slack instance.
- Fact A (known): EDF has unbounded speedup with tool waits.

**Conjectured.**
- EDF needs only O(1/δ) speed for arbitrary releases (C1).
- With known tool durations, some online algorithm has constant speedup (C2).
- With slack, O(1/δ)-competitive weighted throughput is achievable when tool durations are known (C3).
- Family limits for R4 and R5 (C4, C5).

**Open** (Section 9).
- Arbitrary releases with unknown tools: the "lag" obstacle.
- Clairvoyant tools with arrivals and no slack.
- Weighted throughput for multi-GPU-node tasks.
- Memory-competitive KV policies.
- Prefill in the engine model.
- Stochastic durations against a non-clairvoyant benchmark.

**Novelty (honest judgment).**
- Fact A, NP-hardness, the classical online bounds, and EDF's unbounded speedup are known and are not claimed.
- I did not find Thm 2 or Thm 4 in the sources I checked (Section 4).
  - Thm 2 is a short adversary argument. It is natural enough that it may be folklore.
  - Its model differs from the real-time "sporadic task / speedup factor" model. In particular it does **not**
    resolve the open problem stated in the 2019 self-suspension survey.
  - The tight constant 1 + 1/δ (Thm 2 with Thm 4) is the most interesting result here. Its proof is short.
- I would describe the results as a clean, correct, small contribution relative to the literature checked, not
  as a solution of the general problem.

---------------------------------------------------------------------------------------------------------------

## 1. Model

- **Machines.** m identical machines with preemption and migration (by Lemma 2 this is the batched engine up to
  a factor 2 in speed; Section 8). The online algorithm runs at speed s ≥ 1 and the offline benchmark at speed 1.
- **Tasks.** Task j has release r_j, deadline d_j, value w_j, and a DAG G_j of
  - GPU nodes v with work p_v > 0. A node runs on at most one machine at a time and takes p_v/s machine-time at
    speed s.
  - TOOL nodes u with duration t_u ≥ 0. A tool starts the moment it is enabled (all predecessors done and
    t ≥ r_j), ends exactly t_u later, is not sped up, and is unlimited in parallelism.

  Eager issue is justified for the offline optimum by Lemma 3 of general_theory.md. Online it is free, since
  tools have no capacity. Task j earns w_j iff all its nodes finish by d_j.
- **Critical path.** C_j = maximum over paths of G_j of Σ p_v (GPU) + Σ t_u (TOOL), at unit speed.
  **Slack δ:** d_j − r_j ≥ (1+δ)·C_j for every j.
- **Information about tool durations:**
  - I0: revealed when the tool finishes (non-clairvoyant).
  - I1: revealed when the tool is issued.
  - I2: revealed at the task's release (clairvoyant).

  GPU work is known at release unless stated otherwise. Future arrivals are unknown.
- **Feasibility-competitive at speed s:** whenever the offline speed-1 schedule meets all deadlines, ALG at
  speed s meets all.
- **(s, c)-competitive:** ALG at speed s earns at least OPT/c, where OPT runs at speed 1.

**Exact offline optimum** (`offline_feasible_1m`, one machine) [proved].

*Encoding.* A schedule meeting all deadlines exists iff there are reals ρ_v (GPU node release), C_v (GPU node
completion) and σ_u (tool start) satisfying:
- ρ_v, σ_u ≥ r_j, and both are at least the end of every predecessor;
- tool end = σ_u + t_u;
- C_v ≥ ρ_v + p_v;
- every end ≤ d_j;
- Horn's interval condition: for all a ∈ {ρ}, b ∈ {C} with a ≤ b, Σ_{w: ρ_w ≥ a, C_w ≤ b} p_w ≤ b − a.

*Necessity.* Take the actual times of a feasible schedule. Every node is processed inside [ρ_w, C_w], so the
work of the nodes whose windows lie in [a, b] fits in b − a.

*Sufficiency.* Run EDF on the GPU nodes with release ρ_v and deadline C_v. Horn's condition means this EDF run
meets every C_v; this is the classical preemptive single-machine result [known; unverified: Horn 1974]. The
schedule is valid because every predecessor and every tool ends by ρ_v.

*Certificates.* Independent of the classical result, every "feasible" answer used in the tests is certified by
an exact replay (`witness_check`). The encoding was cross-validated three ways
(`test_offline_solver_validated_three_ways`):
- it equals EDF feasibility on tool-free single-node tasks;
- every witness replays;
- it never answers "infeasible" when an exhaustive search over static node priorities finds a feasible schedule.

## 2. Scope relative to docs/general_theory.md (what was simplified, and what transfers)

The open problem as posed simplifies the model of general_theory.md in seven ways. A lower bound proved in a
special case remains a lower bound in the full model whenever the special-case instances are valid full-model
instances and the extra freedoms of the full model do not help on them. A positive result does **not** transfer
automatically.

| # | Simplification here | Lower bounds (Thm 2, 3, 6b) | Positive results (Thm 4, 6, 7) |
|---|---|---|---|
| 1 | Tool-consumption points fixed by DAG edges (doc: a runtime decision) | Transfer: in the adversary instances the tool result is the task's sink, so no consumption point avoids waiting for it | Transfer if the online algorithm uses the DAG of true data dependencies. Removing edges never hurts the offline optimum (Lemma 3 argument), so the offline benchmark gains nothing by choosing a different consumption DAG |
| 2 | No KV preloading | Transfer: the instances have no prompt to preload | Not covered: preloading splits nodes and lets offline move work earlier; Thm 4 applies only if online uses the same split |
| 3 | Decode-only step time a + c·B (no prefill in steps) | Transfer (decode-only instances are valid instances) | Thm 6 inherits Lemma 2's decode-only fluid model; prefill not covered |
| 4 | Fixed (adversarial) tool durations; output lengths known | Cor 2′ is exactly a known-distribution (stochastic) lower bound against the clairvoyant benchmark | Thm 4 holds for every realization, hence for random durations path-by-path; nothing proved against a non-clairvoyant (expected-optimal-policy) benchmark |
| 5 | KV options: keep / evict-recompute only (no swap, no shared prefixes) | Transfer (no KV in the instances) | Thm 7 covers evict-recompute only; swap is an off-GPU delay (like a tool) and is not covered |
| 6 | Every tool call launched immediately (no choice of which calls to launch) | Transfer (each tool is a true dependency of the task) | Valid under Lemma 3's assumptions (uncapacitated, cost-free, timing-independent tools) |
| 7 | Objective: on-time value only (doc also lists weighted tardiness) | Not transferred to tardiness | Thm 4 gives zero tardiness whenever the offline optimum has zero tardiness; nothing else |

## 3. Assumptions table (which result needs what)

| Result | Machines | Task shape | Tool info | Releases | Slack | Speed | Objective |
|---|---|---|---|---|---|---|---|
| Fact A [known] | 1 | chains | any (EDF ignores it) | common | 0 | any s fails | feasibility |
| Thm 2 [proved] | m | GPU→TOOL | I0 / I1 (upper bound known) | common | δ ≥ 0 | needs s ≥ 1+1/δ (δ=0: none suffices) | feasibility |
| Cor 2′ [proved] | 1 | GPU→TOOL | I0 / I1, known distribution | common | δ | P(miss) ≥ 1 − s(δ/(1+δ) + 1/k) | feasibility |
| Thm 3 [proved] | 1 | GPU→TOOL | I0 / I1 | common | 0 | every s: ratio unbounded | unit-value throughput |
| Thm 4 [proved] | m | DAG | none needed (adversarial) | "r_i<r_j ⇒ d_i>d_j" (incl. common) | δ > 0 | s = 1+1/δ suffices (EDF) | feasibility |
| R4 [refuted ext.] | 1 | chains | any | arbitrary | δ = 1 | EDF fails at 2, 9/4, 5/2 | feasibility |
| Prop 0 [proved] | 1 | DAG | I2 | common | 0 | s = 1 (exponential time) | feasibility |
| R5 [refuted] | 1 | chains | I2 | common | 0 | CP-EDF fails at ≤ 29/20 | feasibility |
| Prop 5, Cor 5′ [proved] | 1 (m via known results) | one GPU node + tool DAGs | downstream known | arbitrary | any | classical | feasibility / value |
| Thm 6 [proved, decode-only] | engine | DAG | none | as Thm 4 | δ | engine speedup 2(1+1/δ) | feasibility |
| Thm 6b [proved] | engine | GPU→TOOL | I0 / I1 | common | 0 | every λ fails | feasibility |
| Thm 7 [proved] | m | DAG | none | as Thm 4 | δ | (1+β)(1+1/δ) | feasibility, KV evict-recompute |

## 4. Literature (checked in this session)

- **Ridouard, Richard, Cottet, "Negative results for scheduling independent hard real-time tasks with
  self-suspensions", RTSS 2004** ([PDF](https://www.lias-lab.fr/publications/7082/2004-IEEE-Ridouard.pdf)).
  Read: Thm 1 (feasibility strongly NP-hard, ≤ 1 suspension per task, implicit deadlines); Thm 2 (no
  universal scheduling algorithm unless P = NP); Thm 6 ("EDF is not optimal even with a k-speed processor").
  The Thm 6 instance is τ1: C = 2k, D = 4k−1; τ2: 1 → suspension 4k−2 → 1, D = 4k. Its prose notes that
  suspensions do not shrink with a faster processor. **[known; cite]**, used as Fact A.
- **J.-J. Chen, "Computational complexity and speedup factors analyses for self-suspending tasks", RTSS 2016**
  ([PDF](https://ls12-www.cs.tu-dortmund.de/daes/media/documents/publications/downloads/2016-JJ-Suspension-Complexity.pdf)).
  Read: Thm 5 says the speedup factor of FP, EDF, suspension-aware EDF, EDZL, suspension-aware EDZL and SULLF
  is not bounded by a constant (dynamic self-suspension, suspension not sped up). Sec. 4.3 says the author
  could not construct a speedup lower bound for suspension-aware LLF. Thms 6–7 give constant RM speedup when
  S_i ≤ λC_i or (C_i+S_i) ≤ σT_i (sporadic, utilization-based; a different slack notion). **[known; cite]**
- **Chen, Nelissen, Huang, et al., "Many suspensions, many problems: a review of self-suspending tasks in
  real-time systems", Real-Time Systems 55:144–207 (2019)**
  ([PDF](https://openaccess.city.ac.uk/26150/1/Chen2019_Article_ManySuspensionsManyProblemsARe.pdf)). Read
  Sec. 8.1:
  - segmented DAG model: "there is no known theoretical lower bound with respect to the speedup factors";
  - Ridouard & Richard (2006) proved that no deterministic online scheduler can be optimal for
    self-suspending tasks (reported there; primary source not read: **[known; secondary]**);
  - dynamic self-suspension: "How to design good schedulers with a constant speedup factor remains as an open
    problem".

  **Positioning of Thm 2.** That survey's benchmark is a scheduler that must handle *every* legal suspension
  pattern of a sporadic task system. Thm 2's benchmark is the clairvoyant offline schedule of the *realized*
  instance. Under the survey's benchmark the Thm 2 task set is infeasible (all-long patterns). So Thm 2 does
  not answer the survey's question. It answers the competitive-analysis question for one-shot jobs.
- **F. Eberle, "O(1/ε) is the answer in online weighted throughput maximization", STACS 2024**
  ([LIPIcs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.STACS.2024.32),
  [arXiv 2310.16697](https://arxiv.org/abs/2310.16697)). Abstract read: preemptive, single or unrelated
  machines, jobs with ε-slack (window ≥ (1+ε)·processing time); an O(1/ε)-competitive algorithm for weighted
  throughput, matching the known lower bound for unweighted throughput on one machine. No speed augmentation;
  no suspensions. **[known; cite (abstract)]**
- **Eberle, Megow, Schewior, ESA 2020**
  ([LIPIcs](https://drops.dagstuhl.de/opus/volltexte/2020/12907)). Θ(1/ε) for unweighted throughput with
  slack, and the Ω(1/ε) lower bound. **[known; secondary]** (search snippet and the STACS abstract).
- **JITServe (Zhang et al., NSDI 2026, [arXiv 2504.20068v3](https://arxiv.org/html/2504.20068v3)).** Read
  through the HTML (Sec. 4.2, Appendix D). The guarantee GMAX ≥ (1/8.56)·offline goodput (Thm 4.1; the
  appendix's Thm D.3 gives r ≈ 1/8.557) is proved for:
  - independent requests, each with a computing time, an SLO time and a goodput, earned only if it completes
    by its SLO;
  - analysis of a single service slot ("extension to multiple slots is immediate");
  - preemption with a (1+δ) goodput threshold;
  - a direct charging argument;
  - **no tool waits or multi-stage dependencies in the proof, no KV/memory constraint, no batching step-time
    model, no speed augmentation**.

  Goodput is length-proportional, which bounds value density; that is consistent with the Baruah and
  Koren–Shasha lower bounds. **[known; cite (as read in v3 HTML; proof not checked)]**.

  Our model differs exactly where Thms 2–3 bite. Requests are multi-stage with tool waits whose durations are
  not known in advance. Thm 3 shows that no constant ratio is possible there for any speed, so JITServe's
  bound cannot extend to our model without tool clairvoyance (downstream durations) or slack.
- **Koren & Shasha, D^over, SIAM J. Comput. 24(2) 1995**: optimal ratio 1/(1+√k)² for importance ratio k
  **[known; secondary]**.
- **Baruah et al. 1992** (1/4 bound for value = length): **[known; unverified]**.
- **Phillips, Stein, Torng, Wein, STOC 1997**: EDF on m machines with speed 2 − 1/m: **[known; secondary]**.
- **Horn 1974** (interval condition), and **Liu–Layland / Dertouzos** (EDF optimal on one preemptive machine):
  **[known; unverified]**. These are used only where every instance is also certified by exact replay.

## 5. P1: the impossibility frontier

**Fact A [known; cite Ridouard et al. 2004 Thm 6; Chen 2016 Thm 5]. EDF by task deadline has unbounded
speedup with tool waits.**
- Brief's instance: A = GPU 1, d = 2; B = GPU ε → TOOL 10−2ε → GPU ε, d = 10. EDF at speed s finishes B at
  (1+2ε)/s + 10 − 2ε > 10 iff ε < 1/(2s) (Z3).
- Exact runs at s ∈ {1, 2, 5, 10, 100} confirm the closed form.
- Offline is feasible (exact solver plus witness), and CP-EDF meets the instance even at speed 1.
- Ridouard's instance fails for EDF at speed k ∈ {2, 3, 4, 6} and is feasible at speed 1.

Tests: `test_factA_*`.

**Theorem 2 [proved]. Without tool clairvoyance no speed suffices; with slack δ, speed 1 + 1/δ is necessary.**
Let tool durations be revealed no earlier than issue (I1 or I0). Let m ≥ 1, δ ≥ 0, and s ≥ 1 with s·δ < 1+δ
(any s when δ = 0). Then for every deterministic online algorithm ALG there is an instance with these
properties:
- m·q tasks, all released at 0;
- each task a chain GPU(1) → TOOL;
- common deadline D = q;
- every tool duration in {0, L} with L = D/(1+δ) − 1 (so the upper bound L is known);
- every task has slack ≥ δ;
- the instance is feasible at speed 1 on m machines;
- ALG at speed s misses a deadline.

*Construction.* Choose an integer q with q ≥ 1+δ and q·(1/s − δ/(1+δ)) > 1; this is possible because
1/s > δ/(1+δ). The adversary watches ALG. Each tool issued gets duration 0, except the last of the mq tools to
be issued, which gets L. ALG is deterministic, so this defines one fixed instance and ALG's run on it.

*Online.* The total GPU work is mq and the total processing rate is at most ms, so the last first-node
completes at time ≥ q/s. The long task therefore ends at ≥ q/s + L.
- Since q/s + q/(1+δ) − 1 > q ⇔ q(1/s − δ/(1+δ)) > 1, it misses D.
- If some first node never completes by D, that task misses anyway.

*Offline.*
- Run the long task's GPU node first on machine 1, during [0, 1]. Its tool ends at 1 + L = D/(1+δ) ≤ D.
- Pack the remaining mq − 1 unit nodes: each machine runs q of them back to back, so all finish by q = D.
- Slack: the long task has (1+δ)(1 + L) = D exactly; the short ones have (1+δ)·1 ≤ q.

Z3 proves each inequality (`test_thm2_z3_algebra`, including that a valid q exists iff sδ < 1+δ). The adversary
was run exactly against EDF, FIFO, processor sharing and random static priorities, at m ∈ {1, 2} and nine
(δ, s) pairs. Every run misses; the realized instance has slack; the explicit offline schedule meets all
deadlines; replaying the realized instance reproduces the miss (`test_thm2_adversary_defeats_*`).
Worked example: δ = 1, s = 3/2 gives q = 7, D = 7, L = 5/2. Online ≥ 14/3 + 5/2 = 43/6 > 7. ∎

*Remarks.*
- (a) Thm 2 needs no arrivals and holds even if ALG knows all GPU work and the tool upper bound. The failure is
  purely informational: under I2 with common release, speed 1 suffices (Prop 0).
- (b) It is tight at the threshold: at s = 1+1/δ, EDF meets the adversary's instance
  (`test_thm2_is_tight_*`) and every instance in its class (Thm 4).

**Corollary 2′ [proved] (randomized algorithms; known-distribution model).** Consider one machine, k tasks
GPU(1) → TOOL, D = k. Exactly one task, chosen uniformly at random, has tool length L = D/(1+δ) − 1; the others
have 0, and the distribution is known to the algorithm. Then every randomized online algorithm (I0 or I1) at
speed s misses with probability at least 1 − ⌊s·θ⌋/k, where θ = D − L = kδ/(1+δ) + 1. That tends to
1 − sδ/(1+δ) as k → ∞, and to 1 when δ = 0. The clairvoyant offline schedule always meets all deadlines.

*Proof.*
- Fix ALG's random seed and consider its run R on the all-zero instance.
- On the instance whose long index is i, the run coincides with R until task i's tool is issued, because
  the revealed information is identical until then. So task i's first node completes at its R-time c_i.
- Task i misses iff c_i > θ.
- In R, at most ⌊sθ⌋ first nodes complete by θ. So for every seed, at most ⌊sθ⌋ of the k indices succeed.
- Average over the uniform index, which is independent of the seed. ∎

Exact check: for each deterministic policy tested, the number of failing indices is at least k − ⌊sθ⌋
(`test_thm2_randomized_version_counts_failing_indices`).

**Theorem 3 [proved]. On-time value: unbounded ratio at every speed without tool clairvoyance.** Consider one
machine, information I1 or I0, and unit values. For every deterministic ALG, every s ≥ 1 and every k, there is
an instance (k chains GPU(1) → TOOL, released at 0, D = k) on which ALG at speed s completes **no** task on time,
while a speed-1 schedule completes at least ⌊k/s − 1/2⌋.

*Adversary.* When task i's tool is issued at time τ_i, give it length max(0, D − τ_i + 1/2). Then every task
that ALG issues by D finishes after D, and so does every task issued later.

*Offline.*
- Let τ_(1) ≤ τ_(2) ≤ … be ALG's issue times; tasks never issued count as +∞ and get length 0. On one machine
  of speed s, τ_(j) ≥ j/s.
- Serve the last g = ⌊k/s − 1/2⌋ issued tasks, in ALG's issue order, one after another. The l-th finishes its
  GPU node at time l. It is on time iff l ≤ τ_(k−g+l) − 1/2, which follows from
  l ≤ (k−g+l)/s − 1/2 for 1 ≤ l ≤ g whenever g ≤ k/s − 1/2 and s ≥ 1 (Z3, `test_thm3_z3_algebra`).

Exact runs for s ∈ {1, 2, 4, 8} against four policies confirm: online value 0, offline set of size
≥ ⌊k/s − 1/2⌋, and the offline plan replays (`test_thm3_*`). ∎

Hence no (s, c)-competitive algorithm exists under I0/I1 without slack. Contrast: with slack, Eberle's O(1/ε)
holds for jobs without tools. Tool clairvoyance (or slack) is the missing ingredient, not speed.

**Prop 0 [proved] (clairvoyance with no arrivals is enough at speed 1).** Under I2 with all releases equal, the
instance is fully known at time 0. The online algorithm can compute an optimal offline schedule (in exponential
time, since feasibility is strongly NP-hard by Ridouard et al. Thm 1) and replay it.

Exact check: every Z3 witness on random common-release instances replays exactly
(`test_prop0_*`). ∎

So impossibility at constant speed requires either missing tool information (Thms 2 and 3) or future arrivals.
With I2 and arrivals the question is open (C2). The survey reports that no deterministic online algorithm is
optimal at speed 1 [known; secondary: Ridouard & Richard 2006].

**Frontier summary.**

| tool info | arrivals | slack | what holds |
|---|---|---|---|
| I0/I1 | none | 0 | no speed suffices (Thm 2); value ratio unbounded (Thm 3) |
| I0/I1 | none | δ | s* = 1 + 1/δ exactly (Thm 2 + Thm 4) |
| I0/I1 | arbitrary | δ | s ≥ 1+1/δ needed; EDF needs > 5/2 at δ = 1 (R4); upper bound open (C1) |
| I2 | none | any | speed 1 suffices information-theoretically (Prop 0); CP-EDF fails at 29/20 (R5) |
| I2 | arbitrary | 0 | no optimal online at speed 1 (reported); constant speed open (C2) |
| one GPU node, downstream tools known | arbitrary | any | exactly classical deadline scheduling (Prop 5) |

## 6. P2: feasibility under augmentation

**Theorem 4 [proved]. Global EDF at speed 1 + 1/δ is feasibility-competitive for DAG tasks with tool waits
under slack δ, when no earlier-released task has an earlier deadline.**

*Statement.* Assume:
- m machines and arbitrary DAG tasks;
- tool durations arbitrary, even chosen adaptively by an adversary;
- EDF uses only task deadlines (no knowledge of tool durations or GPU work);
- slack δ > 0;
- every pair of tasks satisfies r_i < r_j ⇒ d_i > d_j. This includes the case of all tasks released together.

Let EDF run, at each instant, the m ready GPU nodes of highest priority, with priority given by (d_j, j). If a
speed-1 m-machine schedule meets every deadline, then EDF at speed s ≥ (1+δ)/δ meets every deadline.

*Proof.* Suppose not. Let k be a task that misses, chosen with the smallest (d_k, k) among the tasks that miss.
Let S be the set of tasks with priority (d_j, j) ≤ (d_k, k).
- (i) *Every task in S is released no earlier than r_k.* If j ∈ S had r_j < r_k, the hypothesis would give
  d_j > d_k, contradicting j ∈ S. So every j ∈ S has [r_j, d_j] ⊆ [r_k, d_k], and the offline schedule
  processes all of S's work inside [r_k, d_k] on m unit machines:
  W(S) := Σ_{j∈S} Σ_v p_v ≤ m(d_k − r_k).
- (ii) *Path lemma.* k is unfinished at d_k. Build a path v_1 → … → v_q in G_k as follows. Let v_q be an
  unfinished node all of whose predecessors finished by d_k. Repeatedly let v_{i−1} be a predecessor of v_i
  whose finish time equals v_i's enabling time e(v_i) := max(r_k, latest predecessor finish), as long as
  e(v_i) > r_k. Stop at the first node whose enabling time is r_k. The
  intervals [e(v_i), e(v_{i+1})], with e(v_{q+1}) := d_k, partition [r_k, d_k], and v_i is enabled and
  unfinished during its interval.
  - If v_i is a tool, its interval has length at most t_{v_i}: tools start at enabling and are never delayed.
  - If v_i is a GPU node, then at every instant it is either running, for total time ≤ p_{v_i}/s, or not
    running. If it is ready but not running, EDF is running m nodes of priority ≤ (d_k, k) on all m machines.

  Let Y be the set of instants in [r_k, d_k] at which all m machines run nodes of S. Then
  d_k − r_k ≤ Σ_tool t + Σ_gpu p/s + |Y| ≤ C_k + |Y| (using s ≥ 1 and that the path weight is ≤ C_k).
  So |Y| ≥ (d_k − r_k) − C_k ≥ (d_k − r_k)·δ/(1+δ).
- (iii) *Counting.* During Y the online schedule does m·s·|Y| work on S. Nothing of S is processed before r_k,
  and k is unfinished, so m·s·|Y| < W(S) ≤ m(d_k − r_k). Combined with (ii):
  s·δ/(1+δ) < 1, i.e. s < (1+δ)/δ. This contradicts the choice of s. ∎

Z3 proves the path step and the counting step, and shows the system becomes satisfiable once the speed
hypothesis is dropped (`test_thm4_z3_algebra`).

*Property tests.*
- 3000 planted instances: 1–6 random DAG tasks, m ∈ {1, 2, 3}, δ ∈ {1/4, 1/2, 1, 2, 3}, common release or the
  release-order condition, mostly at minimum slack. Each is feasible by construction; the deadline is the
  finish time of a random speed-1 schedule, raised to (1+δ)C if needed.
- 300 random non-planted single-machine instances checked against the exact offline optimum.
- The Thm 2 adversary at threshold speed.

All pass (`test_thm4_*`). A local search that tried to maximize EDF lateness at threshold speed found no
violation (scratch only).

*Remarks.*
- EDF needs neither tool durations nor GPU work, so Thm 4 holds under I0. Together with Thm 2 it pins the
  speed–slack trade-off for this class exactly: s* = 1 + 1/δ.
- On one machine without tools, EDF is optimal at speed 1, so the content of Thm 4 is entirely about tool
  waits.
- Offline speed σ (online speed σ(1+δ)/δ, slack measured with GPU times p/σ): rescale work by 1/σ. The proof is
  unchanged.

**R4 [refuted]. "Thm 4 holds for arbitrary release times."**

*Instance* (`r4_general_release_counterexample`): one machine, δ = 1 (so 1+1/δ = 2), EDF at speed 9/4, with
ε = 1/10 and y = 12.
- Z = GPU(27), r = 0, d = 54.
- A = GPU(1/10) → TOOL(27) → GPU(119/10), r = 0, d = 78.
- K = GPU(1/10) → TOOL(99/5) → GPU(1/10), r_K = 1757/45 (= y + ε/s + 27), d_K = r_K + 40.
- N1, N2 = GPU(39/2), r = r_K, d = r_K + 79/2.

Every task has slack ≥ 1; A and K have exactly 1.

*Online EDF at 9/4.*
1. Z runs in [0, 12], because d_Z < d_A.
2. A's first node finishes at 12 + 2/45, so its tool ends exactly at r_K.
3. At r_K, EDF runs A's second node, then N1, then N2, all with deadlines < d_K, for 50.9/(9/4) time.
4. K's first node only starts at ≈ 61.67. K finishes at 734/9 ≈ 81.56 > d_K ≈ 79.04.

*Offline at speed 1.*
1. A's first node [0, 0.1], then Z [0.1, 27.1].
2. A's second node [27.1, 39.0], which is before r_K.
3. K's first node at r_K; N1 and N2 around K's last node.
4. All tasks are on time. The explicit static-priority replay and the exact solver with witness both confirm
   it.

*Same family at other speeds.* EDF also misses at speeds 2, 21/10, 23/10 and 12/5. Another member
(`r4_at_5_2`) makes EDF miss at speed 5/2.

*Why it fails.* The hypothesis of Thm 4 is violated: A is released before K but has an earlier deadline. A's
second node is "old" work that offline finished before r_K, but that online could not run earlier, because Z
delayed A's tool. This **lag** is exactly where the proof of Thm 4 breaks (Section 9, O1). CP-EDF, which knows
tool durations, meets this instance at 9/4. (`test_r4_*`)

**R5 [refuted]. "CP-EDF (EDF on node latest-completion times LCT(v) = d_j − tail(v), tool durations known) is
optimal at speed 1, or even at speed 29/20."**

*Instance* (`cpedf_two_task_family`): two tasks, common release 0, zero slack.
- T0 = GPU(1) → TOOL(5) → GPU(11/10), d = 711/100.
- T1 = GPU(1/100) → TOOL(49/10) → GPU(11/10), d = 706/100.

*What happens.*
- CP-EDF runs T0's first node first, since its LCT (101/100) is earlier.
- T1's second node therefore becomes ready later and, having the earlier LCT, preempts T0's zero-slack last
  node.
- CP-EDF misses at every tested s ∈ {1, 6/5, 13/10, 7/5, 29/20}.

*Offline.* Running T1's tiny first node first meets both deadlines (explicit replay and exact solver).

*Other evidence.*
- A local search also found a general-release CP-EDF failure at speed 6/5 (`cpedf_search_instance`).
- EDF by task deadline meets the R5 family at every tested s > 1, so the two policies fail on different
  instances.

(`test_r5_*`)

**Conjecture C4 (limit of the R4 family).** With ε → 0, the R4 family defeats EDF iff (D + y)/s > D/2, where the
slack of A forces y ≤ D/(s+1). That gives every s < (1+√17)/2 ≈ 2.56 at δ = 1. Z3 checks this algebra
(`test_r4_family_z3_failure_region`), and the exact member meets its deadline at 13/5. The algebra is a
derivation with ε → 0, not a proof that no member fails beyond 2.56, so it is a conjecture.

**Conjecture C5 (limit of the R5 family).** The R5 family fails iff a > (a+c+u)(s−1). The priority constraints
force a < c, so the family cannot reach s ≥ 3/2 (Z3 for this algebra; the exact member meets at 3/2).

**Conjecture C1.** EDF is O((1+δ)/δ)-speed feasibility-competitive for arbitrary releases. Evidence:
- one level of lag (R4) gains at most a constant factor;
- a heuristic two-level cascade analysis (not written up) gained less than one level;
- random exact searches (694 feasible small instances with general releases, scratch) found no EDF failure at
  1+1/δ.

The searches are weak evidence: random search also never found the Thm 2-type failures below 1+1/δ, which
exist.

**Conjecture C2.** Under I2 (known tool durations) with arrivals, some online algorithm is
feasibility-competitive at constant speed, with no slack needed. Evidence: CP-EDF failures found only at
s ≤ 29/20. Hill-climbing searches (scratch) found none at s ∈ {7/5, 3/2}, or with slack δ ∈ {1/2, 1} at
s ∈ {11/10, 6/5}. Weak.

## 7. P3: weighted throughput under overload

**Lower bounds.**
- Without tool clairvoyance: Thm 3 (unbounded at every speed, unit values) and Cor 2′.
- At speed 1 without tools: Koren–Shasha 1/(1+√k)² [known; secondary] and Baruah et al. [known; unverified]
  carry over (tool-free tasks are valid instances).
- With slack ε, Ω(1/ε) for unweighted throughput without tools [known; secondary].

**Prop 5 [proved] (single GPU node: exact reduction).** Let every task j have exactly one GPU node g, with
arbitrary tool DAGs before and after it. Let I_j be the longest tool path into g (from release) and O_j the
longest tool path out of g. Then, on any number of machines, the schedules meeting j's deadline are exactly
those that process g within [r_j + I_j, d_j − O_j].
- g cannot start before every upstream tool path has elapsed.
- After g completes, the downstream tools need exactly O_j more time, unaffected by the GPU.

So the problem is classical online deadline scheduling of jobs with these windows. The job's release is the
observed time at which its last upstream tool returns, so upstream durations need not be known in advance;
the job's deadline d_j − O_j needs the downstream durations. On one machine, EDF on d_j − O_j (that is,
CP-EDF) is optimal at speed 1. Exact check: CP-EDF feasibility equals the exact offline optimum on 150 random
instances with arbitrary releases and tool DAGs before and after (`test_prop5_*`). ∎

*Frontier for this class.* Thm 2's instances are single-GPU-node tasks with unknown downstream tools. So within
this class, knowing the downstream tool durations is necessary and sufficient for speed-competitiveness.

**Corollary 5′ [proved reduction; results known].** For single-GPU-node tasks with known downstream tools:
- Slack translates: if d_j − r_j ≥ (1+δ)(I_j + p_g + O_j), then the job's window is ≥ (1+δ)p_g (Z3,
  `test_cor5_*`).
- Hence Eberle's O(1/δ)-competitive weighted throughput applies [known; cite (abstract)].
- So do D^over [known; secondary] and m-machine EDF at speed 2 − 1/m [known; secondary].

**Conjecture C3.** Under I2 with slack δ, weighted throughput admits an O(1/δ)-competitive (or
O(1)-speed O(1)-competitive) online algorithm for multi-GPU-node tasks. Not proved. Thm 3 shows I2 (or slack)
is needed. R4 and R5 show the natural feasibility policies already lose constants.

## 8. P4: batching and KV memory

**Theorem 6 [proved, in Lemma 2's decode-only fluid model].**

*Engine model.* With speedup λ, a decode step with B sequences takes (a + c·B)/λ; m := a/c is an integer
(otherwise use ⌊a/c⌋, at a further factor ≤ 2 by Lemma 2). Node v has n_v tokens; set p_v := a·n_v, its
batch-size-1 decode time, and compute C_j and the slack δ with these p_v.

*Statement.* Under the release condition of Thm 4, if the unit engine (λ = 1) meets every deadline, then the
following policy on an engine of speedup λ = 2(1+δ)/δ meets every deadline. Simulate global EDF on m machines
of speed s = (1+δ)/δ in p-units. In each engine step batch exactly the nodes the simulation is running, and
release each node's result (enabling its successors and tools) at its simulated completion time.

*Proof.*
- Lemma 2(ii) (general_theory.md) turns the unit-engine schedule into an m-machine schedule at speed 1/a
  tokens/s, i.e. speed 1 in p-units, with identical node completion times. Tool times depend only on those
  completion times, so they are identical too. Hence the m-machine instance is feasible at speed 1.
- Thm 4 says the simulation meets every deadline.
- The engine runs B ≤ m sequences, each at token rate λ/(a + cB) ≥ s/a, the simulated rate (Z3). So real
  progress is never behind simulated progress. A sequence that finishes early is dropped from the batch,
  which only raises the others' rates. Every result is therefore ready by its simulated completion time. ∎

(`test_thm6_*`)

**Theorem 6b [proved]. The lower bound lifts to the engine.** Under I0/I1 with no slack, for every engine
speedup λ and every deterministic engine policy, the Thm 2 adversary defeats it with k > λ(m+1) tasks of n
tokens each.
- *Online.* Engine throughput is B·λ/(a+cB) < λ/c, so the last first node finishes after k·n·c/λ > n(a+c).
- *Offline.* The unit engine runs the long task alone in n(a+c), then batches the rest;
  D = n(a+c) + n(a + c(k−1)) and L = D − n(a+c).

Z3 checks both inequalities. ∎ With the measured m = 51 and λ = 4, k = 209 tasks suffice.

**Theorem 7 [proved] (evict-on-suspend charges KV to speed).**

*Policy.* A task's KV is discarded while it waits for a tool. Any GPU node with a TOOL predecessor first
recomputes its context, at extra work ρ_v ≤ β·p_v.

*Statement.* Under the hypotheses of Thm 4 on the original instance, compared against an offline schedule with
**unlimited memory and no recompute**, EDF at speed (1+β)(1+δ)/δ on the inflated instance meets every deadline.
The online holds KV only for tasks with an enabled (ready or running) GPU node.

*Proof.*
- At speed 1+β, the offline schedule of the original instance processes each inflated node during the same
  intervals and receives (1+β)p_v ≥ p_v + ρ_v work; results are held until the original times. So the inflated
  instance is feasible at speed 1+β.
- Its critical paths at speed 1+β are no longer than the original ones (Z3), so its slack is ≥ δ.
- Apply Thm 4 with offline speed σ = 1+β. ∎

1000 planted property tests at the worst allowed recompute, β ∈ {1/4, 1, 3}, pass (`test_thm7_*`).

*What this does not give.*
- No memory-competitiveness: tasks with a ready-but-preempted GPU node still hold KV.
- The recompute bound β can be large when long contexts precede short nodes.
- Prefill time is outside Lemma 2's model.

**Prop 8 [proved] (keeping KV under EDF is not memory-competitive).**
- Instance: k = 8 tasks GPU(1) → TOOL(10) → GPU(1), released at 0, d = 12k.
- At every speed s ∈ {1, 2, 4}, EDF suspends all k tasks at once (peak suspended KV = k units).
- Running the tasks one after another (idling allowed) meets every deadline with peak 1 unit.
- So keep-KV EDF needs memory augmentation k, which is unbounded (`test_prop8_*`). ∎

**Prop 9 [proved] (cost-competitive caching gives no deadline guarantee; reviewer point (b)).**

*Instance.* Memory holds one KV.
- A = GPU(1) → TOOL(5) → GPU(1), d = 100.
- B = GPU(1), released at 2, d = 7/2.

*Comparison.*
- Never evicting has recompute cost 0, so it is optimal for the caching objective that Landlord/LRU bounds.
  It forces B to wait for A, so B ends at 5 + 3/s > 7/2 at **every** speed s ≥ 1 (Z3).
- Evicting A at time 2 costs one unit of recompute and meets both deadlines (B ends 2 + 1/s; A ends
  5 + 3/s ≤ 100).
- So a caching policy that is optimal for total recompute cost (a fortiori, c-competitive for any c) composed
  with any GPU scheduler gives no deadline guarantee at any speed (`test_prop9_*`). ∎

**What composes and what does not.**
- *Composes:* speed factors multiply. Batching costs 2 (Lemma 2), evict-on-suspend costs (1+β) (Thm 7), and
  the tool/slack trade-off costs (1+δ)/δ (Thm 4). The combined guarantee is engine speedup 2(1+β)(1+δ)/δ,
  with no memory-competitiveness claim.
- *Does not compose:* recompute-cost competitiveness (Prop 9), and memory under keep policies (Prop 8).

## 9. What remains open, with the exact obstacle

- **O1 (arbitrary releases, I0/I1, slack δ).** Is there an O((1+δ)/δ)-speed online algorithm? Is EDF one (C1)?
  - *Obstacle: lag.* In Thm 4's counting step, tasks released before r_k with deadlines ≤ d_k contribute
    online work after r_k: Λ = Σ (online remaining − offline remaining at r_k). Offline may have finished that
    work earlier because its tools started earlier (R4).
  - A proof needs Λ ≤ c·(d_k − r_k), e.g. by charging each unit of lag to the slack of the delayed task. A
    one-level analysis gives such a bound (R4 is extremal for it), but cascades are not controlled.
  - Natural extensions of the busy-period argument fail because suspended tasks extend the busy window with
    idle time.
- **O2 (I2 with arrivals, no slack).** Is constant speed achievable (C2)? CP-EDF has speedup ≥ 29/20 (R5).
  Chen 2016 could not lower-bound suspension-aware LLF. The survey lists constant-speedup scheduling for
  self-suspension as open, in a different benchmark (Section 4).
- **O3 (P3 beyond one GPU node).** Weighted throughput with tools for multi-GPU-node tasks under I2 and slack
  (C3).
  - *Obstacle:* admission decisions interact with the lag. A task admitted early can be made infeasible later
    by delays that the offline optimum avoids.
  - Value-density rules (D^over) analyze independent jobs only.
- **O4 (memory).** A KV policy that is simultaneously O(1)-memory and O(1)-speed competitive under deadlines.
  - *Obstacles:* preempted tasks hold KV (Prop 8, and the remark after Thm 7), and recompute cost can exceed
    any slack (Prop 9).
  - When tool durations are unknown, the keep/evict decision is ski-rental-like, but the "rent" is deadline
    risk, not cost.
- **O5.** Prefill in mixed steps (Lemma 2 is decode-only), random output lengths, and guarantees against the
  best non-clairvoyant (expected-value) policy rather than the clairvoyant offline optimum.

## 10. Index of machine checks (tests/test_open_problem.py)

| Claim | Tests |
|---|---|
| Offline solver (Section 1) | `test_offline_solver_validated_three_ways` |
| Fact A | `test_factA_edf_misses_brief_instance_at_every_speed`, `test_factA_ridouard_theorem6_instance`, `test_factA_z3_edf_formula_exceeds_deadline_iff_eps_small` |
| Thm 2, Cor 2′ | `test_thm2_adversary_defeats_every_tested_online_policy`, `test_thm2_z3_algebra`, `test_thm2_is_tight_edf_meets_adversary_at_threshold`, `test_thm2_randomized_version_counts_failing_indices` |
| Thm 3 | `test_thm3_online_gets_zero_offline_gets_k_over_s`, `test_thm3_z3_algebra` |
| Thm 4 | `test_thm4_z3_algebra`, `test_thm4_planted_instances` (3000), `test_thm4_exact_offline_random_single_machine` (300) |
| R4, C4 | `test_r4_edf_at_one_plus_one_over_delta_fails_with_general_releases`, `test_r4_member_at_speed_5_2`, `test_r4_family_z3_failure_region` |
| R5, C5 | `test_r5_cpedf_two_task_family_fails`, `test_r5_cpedf_search_instance_general_release`, `test_r5_z3_family_condition` |
| Prop 0, Prop 5, Cor 5′ | `test_prop0_*`, `test_prop5_single_gpu_node_tasks_cpedf_optimal_on_one_machine`, `test_cor5_slack_translates_to_job_slack` |
| Thm 6, 6b | `test_thm6_engine_rate_and_lower_bound_algebra` |
| Thm 7, Props 8–9 | `test_thm7_*`, `test_prop8_*`, `test_prop9_*` |
