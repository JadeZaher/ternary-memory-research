----------------------- MODULE arithmetic_continuation -----------------------
(***************************************************************************)
(* Formal TLA+ Specification of Multi-Step Arithmetic Continuations        *)
(* for State-Space Scratchpad Trajectories in Low-Bit Neural Networks.     *)
(*                                                                         *)
(* Formally specifies operand decoding, carry/borrow register updates,    *)
(* middle-loss continuation supervision, and verified terminal emission.   *)
(***************************************************************************)

EXTENDS Integers, Sequences, Naturals

CONSTANTS
    InitialOperand1,    \* e.g. 15
    InitialOperand2,    \* e.g. 4
    OperationType       \* "SUBTRACTION" or "ADDITION"

VARIABLES
    phase,              \* "DECODE" -> "COMPUTE" -> "VERIFY" -> "DONE"
    accumulator,        \* Numerical value in the Mamba scratchpad register s_t
    carry_borrow,       \* Carry or borrow bit register
    emitted_answer      \* Final decoded token output

vars == <<phase, accumulator, carry_borrow, emitted_answer>>

(***************************************************************************)
(* Expected mathematical result for formal safety assertion.               *)
(***************************************************************************)
ExpectedResult ==
    IF OperationType = "SUBTRACTION"
    THEN InitialOperand1 - InitialOperand2
    ELSE InitialOperand1 + InitialOperand2

Init ==
    /\ phase = "DECODE"
    /\ accumulator = 0
    /\ carry_borrow = 0
    /\ emitted_answer = -1

(***************************************************************************)
(* Phase 1: Continuation Frame 0 loads primary operand into state s_t.     *)
(***************************************************************************)
StepDecode ==
    /\ phase = "DECODE"
    /\ accumulator' = InitialOperand1
    /\ phase' = "COMPUTE"
    /\ UNCHANGED <<carry_borrow, emitted_answer>>

(***************************************************************************)
(* Phase 2: Continuation Frame 1 applies subtrahend/addend to scratchpad.  *)
(***************************************************************************)
StepCompute ==
    /\ phase = "COMPUTE"
    /\ IF OperationType = "SUBTRACTION"
       THEN /\ accumulator' = accumulator - InitialOperand2
            /\ carry_borrow' = IF (InitialOperand1 % 10) < (InitialOperand2 % 10) THEN 1 ELSE 0
       ELSE /\ accumulator' = accumulator + InitialOperand2
            /\ carry_borrow' = IF (InitialOperand1 % 10) + (InitialOperand2 % 10) >= 10 THEN 1 ELSE 0
    /\ phase' = "VERIFY"
    /\ UNCHANGED emitted_answer

(***************************************************************************)
(* Phase 3: Continuation Frame 2 verifies state consistency before emit.   *)
(***************************************************************************)
StepVerify ==
    /\ phase = "VERIFY"
    /\ accumulator = ExpectedResult
    /\ emitted_answer' = accumulator
    /\ phase' = "DONE"
    /\ UNCHANGED <<accumulator, carry_borrow>>

(***************************************************************************)
(* Terminal Phase: Done stuttering.                                        *)
(***************************************************************************)
StepDone ==
    /\ phase = "DONE"
    /\ UNCHANGED vars

Next ==
    \/ StepDecode
    \/ StepCompute
    \/ StepVerify
    \/ StepDone

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(***************************************************************************)
(* Safety & Liveness Theorems                                              *)
(***************************************************************************)
\* Safety: Final emitted answer strictly matches mathematical expectation
CorrectnessSafety == [](phase = "DONE" => emitted_answer = ExpectedResult)

\* Liveness: Arithmetic evaluation is guaranteed to terminate
TerminationLiveness == <>(phase = "DONE")

\* Scratchpad Monotonicity: Scratchpad never loses operand state mid-computation
ScratchpadIntegrity == [](phase \in {"COMPUTE", "VERIFY", "DONE"} => accumulator > 0)

=============================================================================
