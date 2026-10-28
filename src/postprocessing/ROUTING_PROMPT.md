You choose an initial boundary simplification tool for EACH agricultural parcel.
The goal is a compact editable polygon that follows the physical field margin.
Inspect plain original imagery and the actual candidate overlays, not only the
number of vertices. Prefer fewer vertices when real corners/bends are preserved;
do not exchange a visible shape error for a tiny point saving.

The three available tools are nth_point (fixed vertex stride), radial_distance
(distance from last retained vertex), and dp (topology-preserving DP at 1px).
Tool parameters are fixed in this experiment. You choose the tool independently
for every instance. All three actual previews are supplied; do not invent a new
parameter, fourth tool, coordinate edit, or a tool you have not observed.
There is NO requirement to use all three tools. Choosing DP for every instance
is permitted when it best preserves physical boundaries. Equal output paths do
not constitute evidence that a method improved the boundary.

The raw prediction and all candidates can be wrong. Ground-truth annotations,
prior evaluation scores, and results of other GPT runs are not available.
Geometric validity and closeness to the raw prediction do not prove accuracy.
Candidate output may have reverted to raw geometry after a legality check; that
is explicitly reported, and can mean a high vertex count.

Each preview row is one instance: plain / raw / nth / radial / DP. Image_mapping
gives original crop coordinates. All model evidence_point coordinates refer to
the ORIGINAL image, not resized panels. Inspect turns, straight sides, narrow
features and physical field margins; ignore texture or shadow alone.

Return exactly one decision per supplied instance_id, with tool, an approximate
evidence_point and a concise observable reason (at most 25 English words).
No unexplained automatic fallback or forced tool diversity is allowed.
The chosen complete parcel is executed exactly. Subsequent local visual editing
is a separate stage and is evaluated separately; do not assume it will fix an
obvious poor initial choice. Record only methods/evidence actually used.
