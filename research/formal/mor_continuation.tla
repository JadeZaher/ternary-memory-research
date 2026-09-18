------------------------- MODULE mor_continuation -------------------------
(***************************************************************************)
(* Formal TLA+ Specification of Mixture-of-Recursions (MoR) Continuation   *)
(* Interpreters in Native Ternary Neural Architectures.                    *)
(*                                                                         *)
(* Models the CPS execution loop, the defunctionalized KV-cache frame      *)
(* store, the exit router continuation selector (call/cc), and verifies    *)
(* termination liveness and absence of infinite-loop attractor traps.      *)
(***************************************************************************)

EXTENDS Integers, Sequences, Naturals

CONSTANTS 
    MaxLoops,       \* Upper bound on recursion budget M_max (e.g. 6 or 8)
    ExitThreshold   \* Scalar halting threshold (e.g. 70 representing 0.70)

VARIABLES
    loop_idx,       \* Current continuation frame index k in 0..MaxLoops
    h_norm,         \* Norm/energy of hidden state representation
    exit_prob,      \* Exit probability predicted by continuation selector
    terminated,     \* Flag indicating call/cc invoked final exit continuation
    kv_stack        \* Defunctionalized continuation store (stack of frame IDs)

vars == <<loop_idx, h_norm, exit_prob, terminated, kv_stack>>

(***************************************************************************)
(* Type invariant: variables stay within valid state spaces.               *)
(***************************************************************************)
TypeOK ==
    /\ loop_idx \in 0..MaxLoops
    /\ h_norm \in 0..1000
    /\ exit_prob \in 0..100
    /\ terminated \in {TRUE, FALSE}
    /\ Len(kv_stack) = loop_idx

(***************************************************************************)
(* Initial state: Frame 0, non-zero representation, empty continuation.    *)
(***************************************************************************)
Init ==
    /\ loop_idx = 0
    /\ h_norm = 100                 \* Initial non-zero embedding norm
    /\ exit_prob = 10               \* Initial low exit probability (e.g. 0.10)
    /\ terminated = FALSE
    /\ kv_stack = << >>

(***************************************************************************)
(* Recurse Action: invokes continuation step, pushes frame to KV-stack,    *)
(* applies contractive transform (FPF contraction factor rho < 1).         *)
(***************************************************************************)
RecurseStep ==
    /\ ~terminated
    /\ loop_idx < MaxLoops
    /\ exit_prob < ExitThreshold
    /\ loop_idx' = loop_idx + 1
    /\ kv_stack' = Append(kv_stack, loop_idx)
    \* Contractive state update: state norm relaxes toward stationary attractor
    /\ h_norm' = (h_norm * 90) \div 100 + 5
    \* Dynamic exit probability increases as state stabilizes
    /\ exit_prob' = exit_prob + 25
    /\ UNCHANGED terminated

(***************************************************************************)
(* Exit Action: continuation selector chooses k_exit (call/cc escape).     *)
(***************************************************************************)
InvokeExitContinuation ==
    /\ ~terminated
    /\ (exit_prob >= ExitThreshold \/ loop_idx = MaxLoops)
    /\ terminated' = TRUE
    /\ UNCHANGED <<loop_idx, h_norm, exit_prob, kv_stack>>

(***************************************************************************)
(* Terminated Stuttering: execution complete, state frozen.                *)
(***************************************************************************)
Done ==
    /\ terminated
    /\ UNCHANGED vars

Next ==
    \/ RecurseStep
    \/ InvokeExitContinuation
    \/ Done

(***************************************************************************)
(* Temporal Specifications and Verifiable Properties                       *)
(***************************************************************************)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Safety 1: Recursion budget is strictly bounded by MaxLoops
BudgetBound == [](loop_idx <= MaxLoops)

\* Safety 2: State representation never collapses to null vector (Zero-FFN trap)
NonZeroRepresentation == [](h_norm > 0)

\* Liveness: Every token continuation eventually halts and invokes exit
TerminationLiveness == <>(terminated = TRUE)

\* Reynolds Defunctionalization Invariant: KV-stack depth equals recursion index
ContinuationStoreIntegrity == [](Len(kv_stack) = loop_idx)

=============================================================================
