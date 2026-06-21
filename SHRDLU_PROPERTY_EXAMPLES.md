# SHRDLU Property / Specification Examples

This note gives example properties built from the primitive AP style in
`SHRDLU_AP_CANDIDATES.json`.

Important:

- The primitive AP file only contains generic unit facts.
- These example properties may introduce object-indexed or relation-level predicates such as
  `object_7_resting_on_object_3`.
- Those are not new primitive APs in the current reduced vocabulary.
- They are example specification-level facts or derived predicates that a verifier could expose.

## 1. Object 7 Should Never Be Above Object 3

Informal meaning:

- Object `7` should never rest on object `3`.

Specification:

- `G NOT(object_7_resting_on_object_3)`

Equivalent English:

- Always, it is not the case that object 7 is resting on object 3.

## 2. The Grasper Must Never Move While Lowered

Informal meaning:

- A move action should never occur in a state where the grasper is lowered.

Specification:

- `G(last_action_move_grasper -> NOT grasper_tag_lowered_before_action)`

If we model transition snapshots with a pre-state view:

- `G(last_action_move_grasper -> NOT pre.grasper_tag_lowered)`

This corresponds directly to the controller rule:

- "Grasper must be raised before it can be moved."

## 3. The Grasper Must Never Be Opened While Holding and Unsupported

Informal meaning:

- If the last action is `open_grasper` and the grasper was holding an object, that held object
  must already be resting on some support.

Specification:

- `G((last_action_open_grasper AND pre.grasper_tag_grasped_nonnull) -> pre.held_object_resting_on_support)`

This corresponds to the simulator rule that objects cannot be dropped into unsupported space.

## 4. Closing the Grasper Should Make It Closed

Informal meaning:

- After `close_grasper`, the grasper should be closed.

Specification:

- `G(last_action_close_grasper -> post.grasper_tag_closed)`

This is an effect specification.

## 5. Raising the Grasper Should Make It Not Lowered

Informal meaning:

- After `raise_grasper`, the grasper should no longer be lowered.

Specification:

- `G(last_action_raise_grasper -> NOT post.grasper_tag_lowered)`

## 6. Lowering the Grasper Should Make It Lowered

Informal meaning:

- After `lower_grasper`, the grasper should be lowered.

Specification:

- `G(last_action_lower_grasper -> post.grasper_tag_lowered)`

## 7. Highlighting an Object Should Eventually Make It Highlighted

Informal meaning:

- After `highlight_object(obj_i)`, the target object should be highlighted in the resulting state.

Specification:

- `G(highlight_object_i -> post.object_i_tag_highlight)`

If parameterized:

- `G(highlight_object(o) -> post.object_o_tag_highlight)`

## 8. Unhighlighting an Object Should Eventually Make It Not Highlighted

Informal meaning:

- After `unhighlight_object(obj_i)`, the target object should not be highlighted.

Specification:

- `G(unhighlight_object_i -> NOT post.object_i_tag_highlight)`

## 9. A Grasped Object Must Indicate It Is Grasped By Some Grasper

Informal meaning:

- If an object is currently grasped, it should also record that it is grasped by a grasper.

Specification:

- `G(object_tag_grasped_nonnull -> object_tag_grasped_by_nonnull)`

Note:

- In the current reduced AP set, only `object_tag_grasped_by_nonnull` exists.
- A fuller verifier may need a companion AP such as `object_is_currently_grasped`.

## 10. A Lowered Closed Grasper on a Graspable Object May Lead To Holding

Informal meaning:

- If the agent closes the grasper while lowered on a graspable object, the post-state may hold an
  object.

Specification:

- `G((last_action_close_grasper AND pre.grasper_tag_lowered AND pre.grasper_resting_on_graspable) -> post.grasper_tag_grasped_nonnull)`

This is a useful expected-effect property for planning verification.

## 11. Object 7 Should Always Remain Graspable

Informal meaning:

- Object `7` is expected to always have `graspable=True`.

Specification:

- `G(object_7_tag_graspable)`

This is more of an invariant about object metadata than about behavior.

## 12. The White Box Should Always Be Able To Support Objects

Informal meaning:

- The box object should always have `can_support=True`.

Specification:

- `G(object_white_box_tag_can_support)`

Again, this is a scene invariant rather than an action constraint.

## Observations

There are several kinds of properties here:

1. Action precondition properties
   Example:
   - move cannot happen while lowered

2. Transition effect properties
   Example:
   - raise leads to not-lowered

3. Scene invariants
   Example:
   - object 7 is always graspable

4. Object relation invariants
   Example:
   - object 7 should never be on object 3

## Likely Next Step

If we want these properties to be evaluated mechanically, we probably need a second layer of
derived predicates, for example:

- `object_i_resting_on_object_j`
- `object_i_highlighted`
- `object_i_graspable`
- `object_i_can_support`
- `pre.*` and `post.*` views for transition properties

That would let us keep primitive APs small while still expressing interesting specifications.
