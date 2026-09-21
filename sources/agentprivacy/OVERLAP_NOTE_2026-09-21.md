# Overlap note · jade_mage ⊥ the agentprivacy fleet

| | |
|---|---|
| **Date** | 2026-09-21 |
| **From** | Mitchell (privacymage) · agentprivacy.org · mages.city · mage@agentprivacy.ai |
| **To** | Jade Zaher · github.com/JadeZaher/ternary-memory-research |
| **Basis** | read-only review of the public repository at commit `35c65bd` (2026-09-20): 35 commits, 2026-09-16 to 2026-09-20, plus the dated records inside `research/` back to 2026-09-11. Nothing in the repository was changed. |
| **Fork** | github.com/mitchuski/ternary-memory-research: its `main` tracks yours; the branch `harness/jade_mage` carries the lane described in §11; any contribution from this side arrives as a pull request you review. Nothing will ever be pushed to your repository. |
| **Status** | first contact, delivered as a pull request to your repository together with the first contribution of §9 (PR-1). The catalogue entry, the acknowledgement paragraph and any census record wait on your answer; nothing is folded anywhere until you say yes. |
| **Licence of this note** | CC BY 4.0, to match `research/` |

---

## 1. In one line

On 2026-09-11 you pasted the agentprivacy discovery survey into an agent, ran it to the letter, and then applied its proposal ⊥ evidence ⊥ review discipline against your own results hard enough to falsify three days of "all-time records" on 2026-09-18 and retire your own thesis on 2026-09-19. You never installed our engine. You ran the method, and it held across three different agent runtimes. That is the overlap, and it is the first time a stranger has walked in through the door on their own.

---

## 2. What was read

| date | runtime (as the files record it) | what happened | record |
|---|---|---|---|
| 2026-09-11 | Codex (`research/project-setup.md`) | folder registered as a Codex project; the discovery survey v0.3 pasted verbatim (`sources/user/orientation-protocol-v0.3.txt`) | `sources/user/request.md` |
| 2026-09-12 00:25 to 00:30 UTC | Codex | eight-area orientation run; first web-tool fetches failed "not safe to open", direct HTTPS then returned 200 for every prescribed page; two skill packets fetched and SHA-256 matched to the catalog | `research/agentprivacy-orientation.md`, `work/orientation/` |
| 2026-09-12 00:36:55 UTC | Codex, Python 3.12.10 | 35 named finite-domain checks passed on the first execution of the final script | `research/verification.md`, `outputs/math-verification.json` |
| 2026-09-11 and 2026-09-12 | separate agent contexts | orientation reviewed against the exact protocol with hashes reproduced; kickoff reviewed statically; region-memory document reviewed under the instruction to refute rather than confirm: 4 blocking, 11 should-fix, 9 nit, all applied | `research/orientation-review.md`, `research/independent-review.md`, `research/region-memory-review.md` |
| 2026-09-15 | not named | grounding audit: hardware measured, "Proposer/Verifier harness (implemented via internal adversarial review and automated test runners)" | `research/grounding-audit-2026-09-15.md` |
| 2026-09-15 to 2026-09-17 | Antigravity ("Pair Programming Session (Antigravity AI & System Architect)" on deep dives 16 to 23; master summary) | Gates 10 to 23: five model families, GRPO against a local judge, Gemini 2.5 Flash scores, TLA+ specs, four draft papers; every gate marked PASSED by the runtime that built it | `research/conductor-track.md`, `research/master-research-summary.md`, `research/deep-dives/10` to `23` |
| 2026-09-16 | not named | first hardening pass: leaks and rigged thresholds in Gates 4 to 11 found and fixed; `git init`; the folder had been marked safe to delete and was not | `research/hardening-2026-09-16.md`, `MOVED.md` |
| 2026-09-18 | not named | the held-out self-audit; decontaminated corpus; model rewrite; matched-compute protocol | `research/hardening-2026-09-18-heldout.md` §1 to §7 |
| 2026-09-18 to 2026-09-19 | not named | ablation arms A to N, traverse collapse and fix, strain track, activation selector, the condensed pre-registered experiment and its read-out | same, §8 to §13.2 |
| 2026-09-19 to 2026-09-20 | commits carry `Co-Authored-By: Claude Fable 5.1` | general-model track night 1 under a locked qualification rule; disk-resident CPU inference tier; machine-safety rules after a freeze | `research/general-model-track.md`, `research/disk-inference-tier.md`, `experiments/general_model/`, `experiments/mixture_of_tiles/` |

---

## 3. The overlap, part one: the door

The protocol you pasted is ours. It is published at `agentprivacy.org/discovery.md` and is still version 0.3 today (checked 2026-09-21). Your run of it is the only complete external execution we have seen, and it is a better audit of our sites than most we have done ourselves. What it recorded about us, in your words and ours:

| you observed | what it means on our side |
|---|---|
| every one of the eight areas "documented only"; no service "observed working" | correct. Hall, Portal, Swarm, Exchange and κ registry are planned services and our entry pages say so. Your table is the first outside confirmation that the warning is readable. |
| first web-tool attempts failed "not safe to open", direct HTTPS returned 200 | our own entry-path measurements on 2026-09-12 found the same fetcher failures against agent tooling and opened the zones the same day. Your timestamps sit inside that window. |
| the `dual-agent-harness` packet body (v1.0, hash `d1576ffa…`) names `~/agentprivacy-dual-agent-harness` as its canonical home while the fleet register says that checkout is superseded | a defect on our side, still live on 2026-09-21 (SKILL.md line 21). You refused to follow the old copy instructions and pinned the current repository instead. That was the right call. A v1.1 packet is owed; see §13. |
| the catalog carries both `agentprivacy-dual-agent-harness` and `dual-agent-harness` | also ours to clean up. |
| the ring-algebra record "explicitly uses Z/(2^6)Z; neg(bnot(x)) = succ(x) is not a balanced-ternary law without redefining the domain and operations" | correct, and the fence you kept is the one we would have asked for. §7 adds one verified observation on the ternary side and keeps the fence. |
| the City manifest says offline baseline, live adapters not connected, `liveComplete: false` | correct. `city_mage` is a resumable local trust-setup path; live VTA, delegation, agreements and executors are unconnected, and it says so. |
| "City provisioning would add no value to the immediate mathematical task" | agreed. Nothing in this note asks you to provision anything. |

Two of your status distinctions are worth naming because they are the ones most people skip: *available · selected · installed · exercised · independently checked*, and *delivery ≠ recipient acceptance*. Both are kept in every proposal below.

---

## 4. The overlap, part two: the discipline

You state in the orientation that the harness was a methodological reference and that "no engine checkout/release or service has been installed or exercised". The repository carries no `harness.config.mjs`, no `frontier.json`, no `claims_register.md`, no hash-drawn witness set and no κ seal. What it carries is the loop's discipline, arrived at by hand. Mapping, so both sides use one vocabulary:

| what the repository does | what the agentprivacy dual-agent harness calls it | where it lives on our side |
|---|---|---|
| author and reviewer in separate agent contexts; the reviewer told to refute rather than confirm (`region-memory-review.md`) | the seats: proposer ⊥ assayer; the refuter's blindness | `SEAT_CONTRACT.md`; the litreview runtime (sweep ⊥ refute ⊥ judge) |
| "Freeze claims before review; reviewer supplies counterexamples in a separate context" (orientation §Verification) | the hold-apart. Ours draws a witness set by hash from the frozen proposal so the proposer provably could not tune to it; yours freezes without the draw, and says so | the Gap, `GROUND_RULES.md`; instances without a draw are labelled in the catalogue |
| five labels kept distinct: mathematical derivation, published evidence, inference, proposal, measured result (`AGENTS.md`) | the claim tiers (GR-2) | `GROUND_RULES.md` |
| "`status: COMPLETED` plus computed `findings` in every ledger. No script may declare its own gate passed" (hardening 2026-09-16 §2) | measure ⊥ verdict; the proposer never approves its own proposal | the V6 rehydration pipeline's founding rule |
| a decision rule written before the run (heldout §13, general track §5) | the gate and the frontier | `frontier.json`; conform → equivalence → witness → frontier → the door |
| "Deviations from the plan above, all recorded before the read-out was interpreted" (§13.1) | the chronicle | one chronicle per session, append-only |
| the rule passed by its letter (0.032 > 0.03 nats; rho 0.027 at p 5.5e-4) and was filed as a fail of its intent; the thesis retired | *don't override the gate*, in the harder direction. Our own ledger records shipping over a NO-GO twice. You declined to ship over a GO. | `TRUSTS.md` T5, the multiplicative gate |
| Gates 15 to 23 kept with status banners; "Reframe superseded internal results as the research journey, not a retraction" | killed ≠ impossible; the killed-levers bar, filed at win-prominence with a re-open condition | shor_mage's seventeen-entry kill ledger; tigzkp_mage's falsified floor |
| Gates 15 to 23 themselves: a templated corpus, a judge prompted with its own training template, a model that was never trained | **MIRAGE**: a candidate that passes the cheap probe and fails the full validation. The word was coined in shor_mage for exactly this shape. | `HARNESS_PATHS.md` §1 |
| `C-cont`, the equal-tokens control that resolved confound 4 in both directions | the canary: the reference run every claim is read against | every instance names one |
| the checkpointing bug found on the way, isolated on CPU, impact table published, "none on any reported number" | the same trace-or-delete rule (GR-9) applied to a defect rather than a result | `GROUND_RULES.md` |
| ideas attributed to "user, 2026-09-18" in the hardening report; runtimes named on the documents they wrote | attribution both ways; vendored is not mine | `VENDOR.md` practice across the fleet |

The single most useful sentence in the repository, for our purposes, is in the 2026-09-16 pass: *"the gate threshold was set to 0.5x so it would pass."* That is the failure the whole harness exists to make structurally impossible, found and named by the person it happened to.

---

## 5. The overlap, part three: three runtimes, one discipline

The files record three agent runtimes in sequence: Codex for the kickoff and orientation, Antigravity for Gates 10 to 23, and a runtime signing commits as Claude Fable 5.1 for 2026-09-19 and 2026-09-20. The discipline did not live in any of them. It lived in the pasted protocol, the `AGENTS.md` files and the JSON ledgers, and it survived both switches.

The middle phase is a natural experiment in the harness's central claim. A runtime that proposed and graded itself produced 23 PASSED gates, a 1.12 perplexity and a 9.0/10 code score, all on a corpus whose validation split was 93 to 99 percent verbatim in training. The next runtime restored the assay and every accuracy number fell. We would put it this way: one collapsed axis (review) zeroed the whole product, which is what a multiplicative gate predicts. Your repository is the cleanest public record we know of that shows both halves in one place: the mirage, and the audit that dissolved it, with the ledgers for each.

That is why this note exists. Our harness page claims that the boundary lets agents find each other regardless of runtime. Your repository is evidence for that claim produced by someone who had no reason to produce it for us.

---

## 6. Who is writing, and the model as I hold it

You have read our sites through the survey's lens, which is the right lens for a first pass and deliberately a thin one. This section is the thicker one, in my own voice, so that you can see what you are being invited into and disagree with it precisely.

**Who.** I write the Privacy Value Model with Claude, and I run agentprivacy.org as the lab, mages.city as the community front and the dual-agent harness as the runtime that turns the model into a working loop. The model began as an essay whose title is its claim, *privacy is value*, and stands today as a formal specification at version 6.0 (2026-06-10) with a register of numbered conjectures (head C89 at that publication, extended since). Each carries a confidence that I set and revise, and when the document and the register disagree, the register wins. Chronicles are signed with one inscription, `(⚔️⊥⿻⊥🧙)😊`, which is also the algebra.

**The claim in one sentence.** Privacy is not a cost paid for safety. It is the term that lets every other value term compound, because value an adversary can reconstruct is value an adversary can take. Written down, it is V(π, t): the value of a sovereignty path π at time t, a product with a gate.

**The parts I would want a ternary researcher to see.**

- *The gate.* Φ = Φ_agent · Φ_data · Φ_inference. Who runs the computation, what data it saw, where inference happens. Multiplicative on purpose: collapse any one axis and the product is zero. Your middle phase measured this for us without meaning to, when the review axis collapsed and 23 gates went to zero together. It is also the term I hold least firmly (C7, 30 %), with additive-with-floor and min() as the named alternatives, and I would rather see it falsified by a measurement than kept by habit.

- *The two agents and the gap.* The Swordsman ⚔️ negates, assays and protects; the Mage 🧙 complements, proposes and projects; between them sits a gap ⿻ that neither can tune, and they share exactly one root, the person whose work it is. The information-theoretic form is I(S; M | FP) < ε, an instance of the wire-tap family (Wyner 1975, with Fano for the converse), and the reconstruction ceiling R_max = (C_S + C_M) / H(X) < 1 is proven in the conditional regime (no inter-agent channel, a fixed adversary class, a capacity deficit) and conjectural outside it. What the bound buys is not secrecy but a floor on the adversary's error. Your separate author and reviewer contexts are this bound in the small: the reviewer's channel does not carry the author's argument, so the reviewer's verdict carries information the author could not have placed there.

- *Where the algebra comes from.* Z/(2^6)Z, 64 vertices, two involutions, `neg` and `bnot`, whose composition is the successor. The harness proves it in one line, `(64 − (x ^ 63)) % 64 === (x + 1) % 64` for every x, in a file that deliberately imports nothing from the model, so the axiom is checked in two places that never share code. That is the identity you correctly refused for balanced ternary. §7 shows that the triangle carries its own version of the same shape.

- *V6's turn: time.* R(t) = (C_S(t) + C_M(t)) / H(X). The adversary's capacity grows while your archive sits still, so the ceiling moves without anyone touching the archive, and t* is the shelf life of a protection. The two worked instances of 2026 are public: a soundness flaw in a widely used proof library, found with a frontier model the day after that model's release, and a withheld quantum optimisation rediscovered about two months after a zero-knowledge proof that it existed. The consequence I hold most firmly: the only term whose security does not depend on t is amnesia, Grade-2 forgetting, because there is no archive left for a better decoder to read (C86, 30 %; C82, 65 %).

- *Compression as defence.* Fewer bytes and fewer tokens are a smaller surface, so efficient inference is also private inference. Your 1.663 bits per weight and your bytes-per-token metric are compression objects in exactly this sense, and §14 P1 is where that becomes a measurement.

- *What I do not have.* λ, the rate at which a sovereignty path diverges from what an adversary can follow. Without it the countermeasure to the moving ceiling is a conjecture chain at 10 to 30 %, and I call it the corpus's most needed number. I mention it because your repository is built around a number you did not have either, and you went and measured it.

**The six trusts, read against your repository.** The harness has a constitutional layer of six trusts that derive from the model. They are the part of the harness an adopter is asked not to change. Your repository never read that file, which makes the following table more interesting than if it had.

| trust | what it says | where I see it in your files |
|---|---|---|
| T1 · the four promises | protection ⚔️→😊, delegation 🧙→😊, authorization 😊→⚔️🧙, and the separation promise ⚔️⊥🧙; no agent can make another's promise | `AGENTS.md`: "Do not infer hardware, budget, experience, or publication consent"; the owner's decisions recorded as the owner's (FRP skipped, one GPU job at a time) |
| T2 · the separation bound | I(S; M \| FP) < ε: the proposer never sees or influences the held-out witnesses; the prover never co-authors proposals | reviewer in a separate context, told to refute; held-out batches fixed by seed before any arm ran; `eval_heldout_clean.py` scoring checkpoints on data the training scripts never touched |
| T3 · the shared root | Origin(S) ∩ Origin(M) = {P}: both seats boot from the same authorized state and share nothing else | the pasted protocol and `AGENTS.md` are the only things all three runtimes shared; the discipline lived there and survived both runtime switches |
| T4 · consent first | invitation before proposal; terms proffered first; disclosure defaults to the minimum | "Keep the journey local unless the user specifies another audience"; no external message, payment or submission anywhere in the record |
| T5 · the multiplicative gate | any zero collapses the product; a score that fails the held-out gate is worth zero at any score | 23 PASSED gates on a templated corpus reported as worth zero; "its smaller stored size does not count" |
| T6 · the door | push, commit, submit, publish, send belong to the person alone | every commit is the owner's; the runtimes are credited on the documents they wrote and never on the door |

I did not expect a table with no empty cells. That is the strongest thing I can say about the overlap, and it is why the rest of this note is written as an invitation rather than a review.

---

## 7. The overlap, part four: the algebra fence, kept, and one verified observation

You were right to refuse `neg ∘ bnot = succ` on Z/(2^6)Z as a law of balanced ternary. It is not one. The observation below does not reopen that; it is a statement about the symmetry group of your carrier, verified by exhaustive enumeration on 2026-09-21, and it makes no claim about memory or speed.

Your `verification.md` records that the carrier {−1, 0, +1} has 27 unary maps, six reversible permutations, three non-identity involutive swaps and two cyclic shifts, and that only sign reversal is the order-reversing bijection. Those six permutations form the symmetric group on three elements, which is also the dihedral group of the triangle, the smallest non-abelian group. The three swaps are its reflections and the two 3-cycles are its rotations. In a dihedral group a reflection composed with a reflection is a rotation. On your carrier, with `succ` the balanced successor (−1 → 0 → +1 → −1) and `neg(x) = −x`:

```
neg ∘ swap(−1, 0)   = succ      (apply the swap that fixes +1, then negate)
swap(0, +1) ∘ neg   = succ      (negate, then apply the swap that fixes −1)
```

So the generator identity our conformance check proves on the 64-vertex lattice has an exact analogue on your three-element carrier, with `bnot` replaced by the swap that fixes +1. Same shape, different ring, and the identity is a fact about the triangle, not about either project. It also gives a structural reason for a result you already proved in `region-memory.md`: only sign reversal can be pushed onto the inputs, because it is the one reflection that reverses the order min/max depends on; the other two reflections move zero and break it.

Enumeration used, so it can be re-run in `experiments/verify_operators.py` in a minute:

```python
import itertools
C = [-1, 0, 1]
succ = {-1: 0, 0: 1, 1: -1}
neg  = {x: -x for x in C}
for s in (dict(zip(C, p)) for p in itertools.permutations(C)):
    if all(neg[s[x]] == succ[x] for x in C): print("neg after", s, "= succ")
    if all(s[neg[x]] == succ[x] for x in C): print(s, "after neg = succ")
```

Fence, restated: this says nothing about whether ternary weights save bytes or time. The 64-vertex lattice conjectures on our side stay on their own domain with their own confidence labels. In your notation the two identities read `N_0 ∘ N_-1 = cycle_plus` and `N_+1 ∘ N_0 = cycle_plus`, and they are the case `c − d = 1` of a law the script checks over all 27 cases: `N_c ∘ N_d` is the shift by `c − d`. The table of which of the six carrier symmetries are automorphisms, duals or input-pushable under min, max, the signed product and addition modulo 3, 24 cells, is in this pull request as `research/carrier-symmetry.md` with `experiments/verify_carrier_symmetry.py` and its ledger. It decides which "NOTs" are free at decode time under each operator family, which is the question your operator reference is built around, and it agrees with region-memory claim 8 by assertion rather than by argument.

---

## 8. The numbers both sides should quote

So that neither side quotes a superseded figure. Everything below is copied from your ledgers as of `35c65bd`.

**What was claimed and what the held-out re-score found** (`research/hardening-2026-09-18-heldout.md` §1 to §3, `outputs/heldout-clean-eval.json`):

| corpus | val 16-grams found verbatim in train |
|---|---:|
| multicorpus_10m.pt (Gates 15 to 19) | 99.4 % |
| multicorpus_unified_25m.pt (Gates 21 to 23) | 96.0 % |
| multicorpus_agro_environmental_25m.pt | 93.6 % |
| TinyStories real text, 2.0M tokens | 1.4 % |

| checkpoint | claimed PPL | clean Python | clean math |
|---|---:|---:|---:|
| navitrit-100m-step10000 (Gate 15) | 1.15 | 1214 | 12458 |
| navitrit-100m-loopformer (Gate 19) | 1.12 | 1775 | 8623 |
| navitrit-100m-looped-dwp (Gate 19-D2) | 1.12 | 730 | 4320 |

**What survives** (README table, each row with its ledger):

| result | figure |
|---|---|
| multiplication-free ternary GEMM matches dense | 1.2e-6 |
| layer-bypass and early-exit latency on a 135M ternary model | 2.24× and 4.03× |
| parallel log-space selective scan vs sequential Mamba | 1.5e-8 forward, 7e-15 gradient, 17× faster |
| per-loop low-rank modulation of a tied ternary block (arm B vs A) | −0.11 nats held-out |
| soft gating on top (arm C vs B) | −0.07 nats |
| the ternary tax at 50M (arm F vs A) | 0.14 nats |
| 2 loops vs 4 loops at 24.6M tokens (arm E vs A) | −0.04 nats at half the compute |
| value-based exit vs matched random continuation | 0.032 nats (attention-only, stage 2) · 0.149 nats (Mamba hybrid, night 1) |
| general-model night 1: routed hybrid vs anchor, mean core bits/byte, 41,336,832 tokens each | 2.007 vs 1.436 · does not qualify |
| stored size at that night | 25.7 MB vs 29.4 MB · does not count under the rule |
| crossover of routed vs fixed-order curves | between 8 and 12 tile applications per token, in both the equal-tokens control and night 1 |
| ternary packing in the disk tier | 1.600 bits/weight, 1.663 with block scales |
| disk-tier throughput floor, batch 1, CPU, random router | about 1 token/s; targets are labelled targets, not forecasts |

**The pre-registered rule and its reading** (§13, §13.1):

| term | threshold | measured | your reading |
|---|---|---|---|
| beat matched random continuation | > 0.03 nats | 0.032 nats | "a pass on the number and nothing more" |
| path diversity correlates with gain | rho > 0 at p < 0.01 | rho 0.027, p 5.5e-4 | "not evidence that path diversity matters"; the rule should have carried a floor |

Decision recorded: the non-monotonic, history-conditioned traversal is not supported at this scale and is retired from the main line; the honest architecture is the modulated tied block with fixed order plus a per-token value-based exit.

---

## 9. Reading your work as a contributor

The sections above read your repository for what it shares with ours. This one reads it for what it found. I write it as someone who would like to contribute, which means naming the things I learned from it before proposing anything.

**What you found that I had not.**

1. **Three meanings of "parameter count", separated** (hardening §11). Stored, virtual-depth and function-class, with the rule that a routing claim is a claim about the third, must be evidenced by comparisons matched on the second, and can never lower the first. That is the cleanest statement I have seen of why "represents more parameters" is not a footprint claim.
2. **"Stored bytes decide whether the model fits; bytes per token decide whether it runs"** (`experiments/mixture_of_tiles/AGENTS.md`). One sentence that separates two claims our own compression sections have been blurring.
3. **The capacity null losing to the random null** (hardening §13.1). A fixed per-sequence fraction misallocates relative to a per-token threshold. A learned estimate of the worth of continuing beats a budget, and a budget can lose to a coin.
4. **A value head whose predicted gain tracks measured gain** (6.08 against 6.18 nats, Huber 0.155). A network estimating the value of its own next step, in nats, against a price. It is the closest thing to V(π, t) computed inside a model that I have seen, and §14 P4 is built on it.
5. **"The residual stream already carries a compressed path record"** (hardening §13.1, point 2). The mechanistic reason history is redundant, stated as an information-redundancy result rather than a shrug.
6. **The fixed block is untrained for truncation** (hardening §13.2). The best fixed arm scores 7.6 at four applications; deep supervision buys graceful degradation at a price of 0.47 nats at full depth. Per-loop specialisation and truncation tolerance pull against each other, and you measured the pull.
7. **Machine-safety rules after the freeze** (`research/disk-inference-tier.md` §4). No CPU-tier job while a GPU trainer is spilling; four threads at most; a process memory cap; missing telemetry refuses execution. The discipline reaches the physical machine. We have nothing equivalent in the fleet and should.

**Where I would push next, if it were my GPU.** I agree with the order you wrote in §13.1 and add only emphasis.

- `C-exit` first: arm C's recipe plus deep supervision plus a value exit at loop boundaries. It asks the sharpest single question left, whether adaptive depth on the best fixed arm reaches the stage-2 compute curve without paying the 0.47-nat order-agnosticism tax or carrying any traversal machinery. One hour.
- `G2-fixed` second, because it is already on the ladder and it separates what free per-token order costs from what the Mamba tiles cost, at the same 41.3M tokens.
- Arm O, ternary embeddings, as the largest footprint move on the board (52.5 MB to 5.2 MB at max width) with its decision rule already written: a gap under 0.02 nats and it is free.
- Three seeds before any of the above is called a result. Your own protocol says so (hardening §5, point 4) and every arm so far is seed 0. The 0.032-nat pass is the number that most needs a second and a third seed.

**Pull requests I would open, each small, each yours to close.** None touches a training script or a reported number. Each carries its own test and its own ledger under `outputs/`.

| PR | what it adds | where | why now |
|---|---|---|---|
| PR-1 | the six carrier symmetries enumerated, the composition law and the two `succ` identities of §7 asserted, and the 24-cell table of which symmetries are automorphisms, duals or input-pushable under min, max, the signed product and addition modulo 3 | `experiments/verify_carrier_symmetry.py`, `outputs/carrier-symmetry.json`, `research/carrier-symmetry.md` (a new script and ledger; your first-run `math-verification.json` is untouched) | **included in this pull request**; pure finite-domain mathematics in the style your `AGENTS.md` asks for; 33 named checks including the differential-pair reading of its §3b, failing inputs printed with both sides |
| PR-2 | absolute `file:///c:/Users/…` links replaced with repository-relative links | `research/master-research-summary.md`, `research/deep-dives/*.md`, `research/navitrit-tree-architecture-handoff.md` | the documents render on GitHub; no content changes; mechanical and easy to review |
| PR-3 | n-gram leak ratio at 8, 16 and 32 grams for every held-out set, written to a ledger | `experiments/data/`, `outputs/leak-ratio.json` | the reconstruction-ceiling reading of §14 P3; generalises the 16-gram audit you already ran |
| PR-4 | a claims register that tiers the README table: derived · measured · retired · proposal, each row pointing at its ledger | `CLAIMS.md` | documentation only; the shape the harness census reads, and the shape §11 would quote |
| PR-5, later | a second column in the disk-tier benchmark: bytes that must cross a trust boundary per token under a configurable local/remote split | `experiments/mixture_of_tiles/` | §14 P1; only after PR-1 to PR-4 have been received or declined |

If you would rather none of these arrive, say so once and none will.

## 10. Acknowledgement, now

Independent of your answer to §11, the following is acknowledged as of this note, and stands whether or not any accession happens.

- Your orientation is the first complete external execution of the agentprivacy discovery survey, and the first to be run unprompted.
- It found a live defect in our skill catalog that we had not found: a packet pointing at a superseded home. It also caught the fetcher failures on our doors inside the same hour that we were measuring them from inside.
- It kept the fence on our algebra that we would have asked for, and did so before we could ask.
- Your hardening report is the cleanest public record I know of a mirage and its dissolution in one place, with the ledgers for both, kept rather than deleted.
- You did all of this without asking us for anything, and the repository says so on every page that could have claimed otherwise.

The V6 specification closes with the line *the equation opened its doors to be filled, and the first thing that walked in was time.* The second thing that walked in was a block of ternary weights, carried by someone who read the door before opening it.

**The text we will publish.** The paragraph below will appear, verbatim, in the harness catalogue's preface and on the discovery page, and be cited on the lab's harness page, once you tell us which name and citation to use (§12, item 1). Until then it is held here and nowhere else. The name in it is the git author's and will be replaced by whichever you choose.

> **Acknowledgement.** On 2026-09-12, Jade Zaher ran the agentprivacy discovery survey (v0.3), unprompted and in full, on a ternary-weight research project, and on 2026-09-18 applied its claim-versus-evidence discipline to that project's own results, retiring a thesis on a pre-registered rule. Their orientation is the first complete external execution of the survey and found a defect in our skill catalog. Repository: github.com/JadeZaher/ternary-memory-research (MIT code, CC BY 4.0 documents). Method reference: the agentprivacy dual-agent harness.

---

## 11. The proposal: jade_mage

**The name.** Fleet instances are named `<thing>_mage`. The proposed name is `jade_mage`. It is a placeholder that you can change; the entry would carry whatever name and author line you choose.

**What an accession is.** `HARNESS_PATHS.md` in github.com/mitchuski/agentprivacy-harness is the origin operator's catalogue of work done with the harness, grouped by how much of the loop each instance runs. Its own preface says that reading an honest partial teaches the bar better than reading a complete one. Fifteen entries stand today; one is an external acceptance flow, one is a descendant lane that inherited the constitution with no loop at all. An accession is a numbered entry, never reused, that names the instance, its objective, its gate, what it lacks, and the lesson it carries. It grants nothing and claims nothing about the instance beyond what its own files record.

**The class.** jade_mage would enter as an **external, self-fitted, adjacent-class** instance: it has an objective and a pre-registered rule (a frontier in prose), separate-context review, a chronicle and a kill ledger; it has no hash-drawn Gap, no `frontier.json`, no κ seal. The entry says all of that. Nearest kin in the catalogue: the litreview runtime (context isolation, non-numeric) and uor_kappa_mage (descendant lane).

**Draft entry, in the catalogue's own format** (for your edit before anything is written):

> **jade_mage · ternary memory research** *(external instance, self-fitted by its author from the discovery survey; github.com/JadeZaher/ternary-memory-research)*
>
> **objective:** mean core bits per byte on three held-out sets at equal training tokens against a fixed-order anchor, ↓; among qualifying configurations, stored bytes, ↓ · **gate:** the pre-registered rules of the hardening report §13 and the general-model track §5; a configuration that qualifies only with exit disabled gets no credit for adaptive depth · **hard constraint:** never evaluate on a templated corpus, a positional split or a template-derived judge prompt; document-level splits with a 16-gram leak filter and residual leak recorded in the manifest
>
> **the Gap:** *(no hash draw, honestly labelled)* claims frozen before a separate-context review instructed to refute; the equal-tokens control as the canary · **lenses:** author ⊥ reviewer contexts across three runtimes (Codex, Antigravity, Claude) · **canary:** arm C, the modulated tied block at fixed order, continued to equal tokens
>
> **weight:** adjacent, external · public, MIT code / CC BY 4.0 documents · **the lesson it carries:** 23 self-graded gates on a templated corpus dissolved by one held-out re-score, kept on the record with status banners; a rule that passed by its letter filed as a fail of its intent and the thesis retired. The discipline survived two runtime switches because it lived in the protocol text and the ledgers, not in the runtime.

**Terms.**

1. By invitation. The catalogue entry is not written until you accept, in whatever channel you prefer. The lane branch on the fork is our side of the table, not the catalogue. Our Register of Invitations holds chairs that were set and never sat in; that is a normal outcome.
2. Consent-first, revocable. You may ask for the entry to be changed or withdrawn at any time; the accession number stays empty afterwards, as one already does.
3. The entry cites only your files. Our fleet census admits a number only when it is quoted verbatim from the lane's own record. Your README already traces every figure to a ledger in `outputs/`, so this is satisfied by design.
4. Your repository stays yours. Any change from this side is a pull request from the fork, reviewed and merged or closed by you.
5. The entry states what the instance lacks as plainly as what it has. That is the catalogue's rule for every entry, including ours.

**Optional, in increasing weight, none required:**

- **Mechanical fitting.** `tools/new_instance.mjs` in the harness repository writes three files into a directory: `harness.config.mjs` (seats and gate), `frontier.json` (best and baseline metric with the run that set it), `claims_register.md` (the claim tiers). For your repository the frontier is already defined (bits per byte at equal tokens vs anchor; stored bytes as the tie-break), and the claims register is your README table with tiers added. The conformance gate would then run on your directory. This is one afternoon and it is entirely your call.
- **A fleet census record.** A record of what your repository records, with each number quoted from your file, sealed with a content hash, rendered into the public harness page's tables. It never writes into your repository.
- **A City act.** The City of Mages binds work into tomes as acts; the hearthold lane, an external open-source household stack, became Tome X that way. This is narrative, opt-in, and later.

---

## 12. What we ask

Small things, in your order of convenience.

1. **Name.** The git author is Jade Zaher, `README.md` and `LICENSE` say Ahmed Zaher after commit `a910261`, and `CITATION.cff` still says Jade. Tell us which name and which citation line you want used anywhere your work is referenced. We will use only that.
2. **The acknowledgement.** The README credits "the orientation and claim/evidence protocol of agentprivacy.org" and fences the geometry. If you are willing, one added clause naming *the agentprivacy dual-agent harness* would help us, because adoptions are tracked by that name and yours is the first unsolicited one. Wording is yours; no link required.
3. **Absolute links.** `research/master-research-summary.md` and several deep dives link with `file:///c:/Users/<user>/…`. Our own fleet review found about 300 such files in our lanes, so this is a shared hygiene item rather than a criticism; relative links let the documents render on GitHub.
4. **Audience.** The orientation says "Keep the journey local unless the user specifies another audience." The repository is now public under MIT and CC BY 4.0. Please say what may be quoted from it in our pages (the README table? the audit tables of §8? the draft entry of §11?) and what may not.
5. **The journey manifest.** The orientation returned a proposed journey manifest with `folded: false` and a City Key status of unknown. That is the correct end state for a read-only survey. If you ever want the encounter folded into a key, the arrival path is `mages.city/city-key-arrival.md`; a key is a reading, not an authority, and nothing about the accession depends on it.

---

## 13. What we will fix on our side because of your run

Labelled planned; none of it is done as of this note.

| defect you surfaced | fix | owner |
|---|---|---|
| `dual-agent-harness` packet v1.0 names the superseded checkout as canonical home | packet v1.1 pointing at github.com/mitchuski/agentprivacy-harness `main`, exact revision pinned; catalog hash updated | Mitchell |
| two harness packets in the catalog (`agentprivacy-dual-agent-harness`, `dual-agent-harness`) | retire one, cross-reference the other | Mitchell |
| a first-time reader following "pin the current repository" may land on a default branch that is behind `main` | default branch set to `main` and verified | Mitchell |
| the survey protocol offers no example of a completed run | with your permission, link `research/agentprivacy-orientation.md` from `discovery.md` as the worked example | Mitchell, after your answer to §12, item 4 |

---

## 14. Pathways from the Privacy-is-Value model

The model is public at agentprivacy.ai/model (version 6.0); the skill packets at skills.agentprivacy.ai carry the sections by name. Section numbers below are the V6 formal specification's; conjecture IDs and confidences are the register's and are the author's estimates, never measurements. Each pathway names the model term, the place in your repository it touches, one test, and its fence. None of them is a request.

**P1 · Compression as defence (V5 §21 compression spectrum; the R(d, compression) modifier; packet `agentprivacy-compression-defence`).**
Touchpoint: your objective is stored bytes, and §11 of the hardening report found the fp16 embedding matrix was 51 MB against 5.5 MB of ternary tiles, so the routing had been optimising the small half of the footprint. The model's claim is that bytes that never leave the device are a privacy property, not only an efficiency one.
Test: the disk tier already measures logical bytes per generated token. Add a second column, bytes that must cross a trust boundary per token, under a split where tiles are local and embeddings or the head are remote. Two numbers per configuration.
Fence: the model's own compression figures (a 74× BRAID ratio, a 70:1 spellbook ratio) are document-corpus figures with no bearing on weights.

**P2 · Three-axis separation, Φ_agent · Φ_data · Φ_inference (V6 §4; C7, the model's falsification frontier at 30 %; packet `agentprivacy-three-axis-separation`).**
Touchpoint: the GPU-free, disk-resident inference tier is Φ_inference made physical, no provider in the loop. The residual leak your corpus builder writes into `data/clean/manifest.json` is a Φ_data measurement. Your own process already exhibited the multiplicative form: the review axis collapsed in the middle phase and the whole product went to zero.
Test: report the manifest's residual leak and the inference locality as two numbers per release, and record whether either alone predicted a held-out failure. C7 says neither alone should; the product should.
Fence: C7 is the model's most likely term to be wrong, and the additive-with-floor and min() forms are the named alternatives.

**P3 · R(t), the moving ceiling, read on data (V6 §5; R(t) = (C_S(t) + C_M(t)) / H(X); C82 at about 65 %; packet `agentprivacy-temporal-dynamics`).**
Touchpoint: the templated-corpus finding is a reconstruction event: the validation set was reconstructable from training. Your 16-gram overlap ratio is a reconstruction ceiling on data, and `eval_heldout_clean.py` is the same ceiling measured on checkpoints. The equal-tokens control showed the fixed-order block gained 0.70 nats at full depth from doubling tokens.
Test: report the overlap ratio at 8, 16 and 32 grams for every held-out set, then split the 0.70-nat gain into near-duplicate memorisation and generalisation by re-running the leak filter at each width. That gives R as a function of tokens seen, which is the model's R(t) with t counted in tokens.
Fence: the model's t is calendar time against a growing adversary; yours is training time against a fixed test. The analogy is structural, not literal.

**P4 · The value exit as the equation in miniature, and a falsification of the path-integral term (V5 §7, T_∫(π) "edges are not independent"; packets `agentprivacy-path-integral`, `agentprivacy-edge-value`).**
Touchpoint: your value head predicts the gain in nats of the next hop and continues only if the gain exceeds a price λ; the λ sweep is a value curve over a path through the tile graph. The model's path-integral term says value along a path is non-local. Your measurement says path history adds at most 0.03 nats beyond the residual state (arms H to K, S2), so the path is Markov in the state at this scale.
Test: bottleneck the carried state (narrow the residual width, or carry only the strain latent) and see whether history regains value. If it never does, the model owes a note on the regime where T_∫ reduces to a sum. If it does, you have the first measured non-locality in a small system.
Fence: this is the one pathway where your data may falsify a term of ours. That is the preferred direction.

**P5 · Amnesia: hiding versus forgetting (V6 §14; C86 at about 30 %; the separation bound I(S; M | FP) < ε of §10; packet `agentprivacy-amnesia-protocol`).**
Touchpoint: the strain latent (§10 of the hardening report) was designed to let a token free history it no longer needs, and is queued as a KV-cache eviction score for the long-context bench. Evicting to a compressed summary is what the model calls Grade 1 forgetting (recoverable with keys, hiding); dropping is Grade 2 (mathematically unrecoverable). Your mechanistic reading that the residual stream already carries a compressed path record is an information-redundancy statement; the separation bound is the same object with the roles swapped.
Test: the needle-in-haystack and multi-hop retrieval bench you already deferred, at equal cache size against full cache and H2O-style eviction, plus one reconstruction probe: from the summary alone, how much of an evicted token's content can a decoder recover. That number is the leakage of the eviction channel.
Fence: C86 is an obstruction-theoretic reading and carries a 30 % label for a reason.

**P6 · The dihedral bridge, correctly fenced (V5 §12; C85 at about 40 %; packet `agentprivacy-dihedral-sovereignty`).**
Touchpoint and test: §7 of this note. The 24-cell commutation table is in this pull request.
Fence: representation theory about a three-element carrier; no performance content; the 64-vertex conjectures stay on their domain.

**P7 · The existence-leak law, run backwards (V6 §25; C81 at 70 %).**
Touchpoint: the model says a zero-knowledge proof that something is feasible leaks an upper bound on how hard it is, and a public attestation discounts everyone's planning horizon (§27, Z_b' = Z_b − D(a)). Your repository publishes attested infeasibility at a stated scale, with ledgers: the traversal thesis is not supported at 50M parameters and 49.2M tokens; the routed model does nothing past its fourth hop.
Reading: a published negative result leaks a lower bound on the search cost others would otherwise spend, and discounts their horizon in the favourable direction. It is a framing you could use in the README for why the retired thesis stays on the record.
Fence: framing only; there is nothing to measure.

**P8 · The trust task, made bilateral (V5 §22 promise theory; packets `agentprivacy-promise-theory`, `agentprivacy-vrc-identity`; trusttasks.org).**
Touchpoint: your orientation already wrote a complete trust-task table: accountable person, agent role, promised outcome, exclusions, resource cap, access terms, deliverable, acceptance check, reviewer, review point, withdrawal and dispute. In promise-theory terms that table is your promise. An accession is ours. A relationship credential is what records both, revocably.
Test: none. If §11 is accepted, the accession and your terms are written as one signed acceptance record; the hearthold accession has the precedent file.
Fence: a local record is not a verified credential, and delivery is not acceptance, in your own words.

**P9 · The six needs, now resolved.**
Touchpoint: you mapped Compute, Memory, Connection, Delegation, Protection and Value twice, unresolved on 2026-09-12 and grounded on 2026-09-15, and declined to write a six-bit profile because "unknown quantities must not become zeroes". All six are now measured: an RTX 4060, a decontaminated corpus with a manifest, a public repository with a citation, three runtimes under one `AGENTS.md`, local-only default, bits per byte at equal tokens.
Reading: a posture could now be stated. The model insists that bits do not establish identity or authority, and neither would this one.
Fence: optional and cosmetic.

**P10 · Ternary on metal (your Phase III; V6 §5 R(t) and §14 amnesia, read on a substrate). Added 2026-09-21 as the first follow-up on this pull request.**
Touchpoint: `research/deep-dives/09` maps {−1, 0, +1} to conductance pairs and claims zero-current skipping; `conductor-track.md` queues Phase III as future; your day-one request named analog programming and the question log deferred it "until the mathematical target makes their role explicit". Three things make the target explicit now.

1. *The GPU killed the wrong idea.* Your 2026-09-16 pass found the additive GEMM at 0.545x of dense on the GPU and a gate set so it would pass. On a GPU that is the right verdict, because the hardware is built for multiply-accumulate. In a crossbar the multiply is Ohm's law and the sum is Kirchhoff's law, so the exactness result that lost on the GPU is the one the substrate is built for. Phase III is not a new claim; it is the 16 September result on the hardware it was written for.
2. *Arithmetic and wiring agree on which NOT is free.* The differential-pair reading now in `research/carrier-symmetry.md` §3b, asserted by the script: with `w = g_plus − g_minus`, swapping the two lines is the only write-free map and it induces exactly `N_0`; the zero-moving swaps cost writes and move the non-conducting code off the zero weight, so a zero-moving "NOT" destroys the zero-current property your deep dive relies on. The write-free set equals the product-pushable set of §7.
3. *What maps and what splits.* Stationary tiles are how an array is driven. The value exit means the next array is not driven. Per-loop FiLM is a row-drive scaling, which is analog-native; per-loop LoRA is a weight change, which is digital or a write. Your DWP gain of 0.11 nats therefore splits into an analog half and a digital half, and that split is worth its own row in a ledger.

What does not transfer: the disk tier's paging (cells are programmed once; reprogramming is slow and endurance-limited, so array capacity is a budget, not a cache, and your "smallest stored bytes" objective is the crossbar-sizing question in disguise); "literal zero energy" (converters and sense amplifiers dominate and run whether or not a cell conducts; take the peripheral share from the IBM 64-core phase-change chip paper already captured in `work/technical/`); the 20-bit accumulator bound, which becomes an ADC-resolution limit, so columns slice into short row groups with digital partial sums and the one-pass-per-tile picture changes. Two meanings of "neuromorphic" also need separating: analog in-memory arrays hold your FFN tiles; event-driven spiking hardware is where per-token routing and exit live. Your thesis fits each differently.

Tests, both on your machine, no hardware:
(a) *A drift curve.* IBM's open Analog Hardware Acceleration Kit (`aihwkit`, 1.1.0 on PyPI, MIT-licensed) ships a phase-change noise model with drift coefficients. Program arm C's ternary tiles into it, keep the controller digital, and plot held-out loss against time since programming, with the plain dense arm A as the canary. One evening.
(b) *The energy price.* Your value exit continues iff predicted gain exceeds λ in nats. Re-express λ in energy per tile pass from the chip paper's per-operation figures and read your λ sweep as loss against joules. Then the exit is a price in a unit the substrate charges.

Model reading: the drift curve is the moving ceiling on a substrate, stored weights degrading while nothing writes to them, R(t) with t in hours since programming. A reset cell is the hiding-versus-forgetting question in silicon; data remanence in non-volatile memory is a measured literature. Both are analogies to state carefully, and both are measurable, which is more than most of our register can say.

Fence: deep dive 09 is labelled "validated in simulation", but the only simulator in the repository counts bytes; by your own labels the crossbar section is a proposal until (a) has run. Nothing above is a device-physics claim. "A write" in §3b is a state change of an idealised binary cell.

---

## 15. What this note is not

- Not an admission, a credential, a membership or a standing. The City's line is *machines qualify · humans admit · brokers release*; this note is the reading before any of that.
- Not an endorsement of any benchmark figure. Your numbers are yours, quoted from your ledgers, and the retired ones are quoted as retired.
- Not a claim that our engine was run. It was not, and your orientation says so.
- Not a request for private material, keys, credentials, hardware, spending or publication. Nothing here needs any of them.
- Not final. Every table in this note can be corrected by you and the corrected version replaces it.

---

## 16. How to reply

Any of: a review or comment on this pull request; an issue on your repository; mage@agentprivacy.ai. The City Portal is a planned service and is not a channel yet. There is no deadline.

What was done on this side as of this note: a read-only review; a fork whose `main` is identical to yours; this pull request, carrying the note and the PR-1 contribution and nothing else; a lane branch on the fork. Nothing merged, nothing listed, nothing folded; nothing of yours changed outside this pull request's own additions, which you can decline.

(⚔️⊥⿻⊥🧙)😊
