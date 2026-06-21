# Plan-Then-Verify Agent Idea

This note captures a possible new agent mode based on the current preplanned agent, but with
verification during planning before any simulator execution happens.

## Goal

Build a new agent that:

- does not execute while planning
- plans one action at a time
- predicts the next simulator state after each planned action
- checks whether that predicted state/action transition violates SHRDLU simulator rules
- backtracks and replans if the predicted action would be invalid
- executes the final action sequence only once the whole plan is verified

## High-Level Difference From Existing Agents

Current modes:

- Step-by-step reactive agent:
  plans one action, executes immediately, observes real new state, repeats
- Preplanned agent:
  plans the full action sequence once, then executes it directly

Proposed new mode:

- Step-by-step planning without execution
- After each planned action, predict the next state
- Verify the predicted step against SHRDLU rules
- If valid, continue planning from predicted state
- If invalid, replan before any simulator execution
- Only execute after the entire plan has been verified

## Proposed Planning Loop

Given:

- user goal/query
- current simulator state

We maintain:

- `current_planning_state`
  This is either the real initial state or a predicted future state.
- `plan_so_far`
  The verified planned actions collected so far.
- `failed_attempts_at_step`
  Previously rejected candidate actions for the current step, along with violation reasons.

### Step A: Plan Next Action

Input to the LLM:

- the overall goal/query
- the current planning state
- optional failed candidate actions for this step
- optional violation reasons for those failed candidates

Output from the LLM:

- either one next action
- or a signal that planning is complete
- or a signal that the goal is not feasible

Important:

- this action is not executed yet
- it is only a candidate plan step

### Step B: Predict Next State

Input to the LLM:

- the current planning state
- the candidate planned action

Output from the LLM:

- predicted next state
- predicted action result / explanation

This predicted state becomes the tentative future state if verification passes.

### Step C: Verify Against SHRDLU Rules

We verify whether the candidate action and predicted transition violate known simulator rules.

Examples of checks:

- moving while lowered
- lowering while already lowered
- raising while already raised
- opening while not closed
- opening while holding an object but not lowered
- dropping onto unsupported space
- closing when already closed
- moving outside allowed bounds
- impossible grasp/drop assumptions

If verification passes:

- append the action to `plan_so_far`
- update `current_planning_state` to the predicted next state
- continue to next planning step

If verification fails:

- record the failed candidate and violation reason
- replan the same step

## Backtracking / Retry Behavior

### Retry At Same Step

For one step:

- propose action
- predict next state
- verify

If invalid:

- try replanning that same step again

After `k` failed attempts for the same step:

- backtrack to the previous planned step
- discard the previous accepted action
- replan that previous step from its predecessor state

This allows recovery if the earlier choice made later progress impossible.

## Completion Conditions

### Successful Plan Completion

If the planner outputs that there is no next action because the goal is complete:

- and all planned steps have passed verification
- then execute the whole plan in one shot on the real simulator

### Infeasible Goal

If:

- a step cannot produce a valid candidate after retry budget is exhausted
- or repeated backtracking still cannot reach a valid complete plan
- or the model explicitly determines the goal is not feasible

Then:

- do not execute anything
- report that the goal appears infeasible under the current simulator rules/state

## Suggested Agent Components

Potential sub-functions:

- `plan_next_action(goal, current_state, failed_candidates)`
- `predict_next_state(current_state, action)`
- `verify_transition(current_state, action, predicted_state)`
- `backtrack_plan(plan_so_far, predicted_states)`
- `execute_verified_plan(plan_so_far)`

## Data We Likely Need

To support this well, we likely want:

- a structured symbolic state representation
- a structured predicted state representation
- an explicit machine-readable verification result
- a record of failed candidate actions and reasons
- a retry budget per step
- a backtracking budget per overall plan

## Important Design Decision

There are two possible verification approaches:

### Option 1: Rule-Based Verification Only

Use explicit Python checks for simulator constraints.

Pros:

- deterministic
- explainable
- aligns closely with real SHRDLU controller behavior

Cons:

- only catches rules we encode
- does not catch deeper semantic mismatch unless encoded

### Option 2: Hybrid Verification

Use:

- Python rule checks for hard constraints
- LLM prediction for soft/anticipated next-state semantics

This is likely the most practical approach.

## Risks / Open Questions

- Predicted state may drift away from the actual simulator state if the model predicts poorly.
- Some SHRDLU rules are local action constraints, but some placement outcomes depend on support
  geometry and object relations that the predictor may hallucinate.
- Backtracking can become expensive if not bounded.
- A full symbolic simulator-side "dry-run" validator would be stronger than pure LLM prediction,
  but would require more code.

## Possible Minimal First Version

A simpler first implementation could be:

1. Plan one action at a time without execution.
2. Verify only hard action preconditions in Python.
3. Use the LLM to predict the next symbolic state.
4. Continue planning from predicted state.
5. Execute only after a full verified plan is built.

This would avoid building a full internal simulator clone while still catching the common failures
like "move before raise".

## Example Failure This Should Prevent

Current preplanned issue:

- plan says `move_grasper(...)`
- but current state says `grasper_lowered=True`
- real execution fails immediately because the grasper must be raised before moving

New agent behavior:

- candidate action: `move_grasper(...)`
- verifier checks current state and sees `grasper_lowered=True`
- verification fails before execution
- planner retries and should propose `raise_grasper()` first

## Naming Ideas

Possible names:

- `VerifiedPreplanAgent`
- `PredictivePlanAgent`
- `BacktrackingPlanAgent`
- `PlanValidateExecuteAgent`

## Recommendation

If implemented, start with:

- Ollama-backed version first
- Python rule-based verification of hard SHRDLU constraints
- LLM-predicted next-state summaries
- bounded retry + bounded backtracking

Then later:

- add OpenAI-compatible version
- improve structured state prediction
- possibly add simulator-side dry-run validation
