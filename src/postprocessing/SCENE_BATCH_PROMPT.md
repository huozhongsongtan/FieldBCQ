# Whole-image parcel boundary simplification

Analyze ALL predicted parcels in the supplied image together. Your goal is to
remove redundant boundary vertices as aggressively as the FIXED quality limits
permit: replace staircase samples and tiny waves along a straight side with its
endpoints; keep only real structural bends. For a simple whole parcel, actively
test roughly 4--8 points; this is an aspiration, not a universal limit. Explain
your visual judgment briefly. GT is unavailable. Gradients are a proxy only.

Every instance has ALL current zero-based ring coordinates in the JSON. The raw
image, full current outlines, ID/vertex-index atlases and joint preview are
visual evidence. Atlas tiles are crops: coordinates in edits MUST remain in the
original image coordinate system. Preserve the fixed shared-edge protection,
identity and validity guards. Do not fill holes, repair overlaps, move vertices
or introduce new detection instances in this vertex-only experiment.

Return exactly the supplied schema: state_id, diagnosis, action, commands,
batch_id, done. ALL decisions address the image, not one parcel at a time.

1. action=propose: issue exactly one command for EVERY unresolved_target_id,
   in a SINGLE response. Each command is direct_edit, use_tool or keep. Use
   batch_id=null and done=false. Direct edits use task=simplify and literal
   retain_vertices indices (>=3, cyclic forward order, fewer than before), or
   replace_boundary_arc with unchanged endpoints and an explicit shorter path.
   Alternatively choose a listed simplifier and explicit numeric parameters.
   Explain each keep; tool failure alone is not a reason to abandon a boundary
   you can simplify through your own geometric judgment. All proposals are
   generated from the SAME state. Local execution does not commit them.
2. Inspect the subsequent individual checks AND the combined full-image preview.
   action=commit: commands=[], batch_id=the observed batch ID. A commit applies
   the ENTIRE observed passing batch atomically after another joint check. Do
   not request a commit of a failed or stale batch. Set done=true only when
   this commit resolves all remaining targets; otherwise false. An accepted
   individual candidate does not imply the combined batch passes.
3. If a candidate/batch fails, action=propose again for every STILL unresolved
   target: revise the offending point list/tolerance, or explicitly keep a
   parcel with a justified reason. This replaces the prior pending batch.
4. action=finish: explicitly keep EVERY remaining unresolved target using keep
   commands and concrete reasons; batch_id=null, done=true. This discards any
   uncommitted preview. Do not claim uncommitted changes were applied.

Normal images should need TWO calls (whole-image proposal, then joint commit).
Reserve at least one later call to observe/commit a proposal. With one call
left, commit an already observed accepted batch or finish with explicit keeps;
do not generate an unseen final proposal. The supplied budget is PER IMAGE.

Prioritize the smallest supported point count under the common limits. Your own
vertex choices are allowed and must not be replaced by an unrecorded optimizer.
All requests, responses, evidence, point lists, tool names/parameters, proposal
rejections, combined checks, commits, keeps and unresolved targets are recorded.
You cannot invent/download/execute external tools in this run.
