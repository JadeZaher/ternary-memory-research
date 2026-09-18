---------------------- MODULE conditional_program_weights ----------------------
(***************************************************************************)
(* Formal TLA+ Specification of Dynamic Weight Parameterization (DWP)      *)
(* and Weights as Conditional Programs for Low-Bit Recurrent Architectures.*)
(*                                                                         *)
(* Formally specifies tile execution roles (BIND, REFINE, VERIFY, EMIT),   *)
(* context-conditioned modulation invariants, PCGrad gradient orthogonal-   *)
(* ization, and termination liveness under non-monotonic routing.          *)
(***************************************************************************)

EXTENDS Integers, Sequences, Naturals

CONSTANTS
    MaxHops,        \* Maximum reasoning hop budget (e.g. 4 or 6)
    NumTiles        \* Number of functional tiles in the module graph (e.g. 12)

VARIABLES
    current_hop,    \* Current execution hop t in 0..MaxHops
    active_tile,    \* ID of physical tile executing at current hop in 1..NumTiles
    active_role,    \* Functional role: "BIND", "REFINE", "VERIFY", "EMIT"
    direction,      \* Hop direction: "FORWARD", "BACKWARD", "SELF"
    has_gradient_conflict, \* Flag indicating whether gradient projection was needed
    terminated      \* Termination flag

vars == <<current_hop, active_tile, active_role, direction, has_gradient_conflict, terminated>>

(***************************************************************************)
(* Valid Roles & Direction Domains                                         *)
(***************************************************************************)
Roles == {"BIND", "REFINE", "VERIFY", "EMIT"}
Directions == {"FORWARD", "BACKWARD", "SELF"}

TypeOK ==
    /\ current_hop \in 0..MaxHops
    /\ active_tile \in 1..NumTiles
    /\ active_role \in Roles
    /\ direction \in Directions
    /\ has_gradient_conflict \in {TRUE, FALSE}
    /\ terminated \in {TRUE, FALSE}

(***************************************************************************)
(* Initial State: Hop 0, BIND role, forward direction.                    *)
(***************************************************************************)
Init ==
    /\ current_hop = 0
    /\ active_tile = 1
    /\ active_role = "BIND"
    /\ direction = "FORWARD"
    /\ has_gradient_conflict = FALSE
    /\ terminated = FALSE

(***************************************************************************)
(* Transition: Bind to Refine (Forward hop)                                *)
(***************************************************************************)
HopBindToRefine ==
    /\ ~terminated
    /\ active_role = "BIND"
    /\ current_hop < MaxHops
    /\ current_hop' = current_hop + 1
    /\ active_tile' = (active_tile % NumTiles) + 1
    /\ active_role' = "REFINE"
    /\ direction' = "FORWARD"
    /\ has_gradient_conflict' = FALSE
    /\ UNCHANGED terminated

(***************************************************************************)
(* Transition: Refine to Verify (Backward hop / Self loop)                 *)
(***************************************************************************)
HopRefineToVerify ==
    /\ ~terminated
    /\ active_role = "REFINE"
    /\ current_hop < MaxHops
    /\ current_hop' = current_hop + 1
    /\ \E t \in 1..NumTiles : active_tile' = t
    /\ active_role' = "VERIFY"
    /\ direction' \in {"BACKWARD", "SELF"}
    \* Potential gradient interference when re-visiting previous tile
    /\ has_gradient_conflict' = (active_tile' <= active_tile)
    /\ UNCHANGED terminated

(***************************************************************************)
(* Transition: Verify to Emit (Terminal hop)                               *)
(***************************************************************************)
HopVerifyToEmit ==
    /\ ~terminated
    /\ active_role = "VERIFY"
    /\ current_hop < MaxHops
    /\ current_hop' = current_hop + 1
    /\ active_tile' = NumTiles
    /\ active_role' = "EMIT"
    /\ direction' = "FORWARD"
    /\ has_gradient_conflict' = FALSE
    /\ terminated' = TRUE

(***************************************************************************)
(* Terminal State Stuttering                                               *)
(***************************************************************************)
StepDone ==
    /\ terminated
    /\ UNCHANGED vars

Next ==
    \/ HopBindToRefine
    \/ HopRefineToVerify
    \/ HopVerifyToEmit
    \/ StepDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(***************************************************************************)
(* Safety & Liveness Theorems                                              *)
(***************************************************************************)
\* Safety 1: Hop budget is strictly respected
HopBudgetBound == [](current_hop <= MaxHops)

\* Safety 2: Role exclusivity - only valid roles executed
RoleExclusivity == [](active_role \in Roles)

\* Safety 3: PCGrad guarantee - whenever conflict occurs, system is in VERIFY/REFINE
PCGradApplicability == [](has_gradient_conflict => active_role \in {"VERIFY", "REFINE"})

\* Liveness: Trajectory always terminates in EMIT role
TerminationLiveness == <>(terminated = TRUE /\ active_role = "EMIT")

=============================================================================
