# momia3 GUI layout studies

Static HTML mockups for the next GUI iteration. They are not implemented UI.

## A · Explorer / Canvas / Inspector

Three persistent columns:

- left: experiment tree, channels, tile thumbnails;
- center: viewer, timeline, filmstrip;
- right: pipeline/results/inspector tabs.

Logic: source, canvas and context stay visible at the same time. Best default for
iterative analysis and for users who switch between several experiments.

Tradeoff: right panel carries both parameters and results; pipeline forms must
stay compact or move to a drawer.

## B · Guided workflow

Left rail is a six-stage workflow with explicit completion state. Center is the
current-stage preview. Right panel contains only the current stage's parameters
and validation. Bottom console keeps the run log visible.

Logic: makes data -> preprocess -> stitch -> segment -> track -> results
explicit, prevents unrelated controls from competing for the same space.

Tradeoff: slower for power users who compare arbitrary operations; data browser
needs to live in the left rail or a secondary tab.

## C · Canvas first

Full-bleed viewer with a narrow icon rail, a collapsible data drawer, a floating
context inspector and a filmstrip. Data and parameters overlay the canvas.

Logic: maximizes pixels for large mosaics and time series; controls are close to
the data without permanently shrinking the canvas.

Tradeoff: overlays hide image regions; less suitable for long parameter forms
or side-by-side table inspection.

## Suggested direction

Use A's persistent three-column information architecture and B's stage/status
logic. Keep C as a focus mode rather than the default shell.

In napari terms: left dock for Dataset/Experiment, center viewer, right dock for
stage-aware Pipeline/Results, bottom dock for Run log. The current single
right-side QTabWidget should be split into those regions.

## Button visual language

- Default text/icon buttons are borderless tonal controls: use the surface-alt fill, no outline, and a brighter fill on hover.
- Primary actions use the accent fill; destructive hover states may tint toward danger, but do not add a permanent red border.
- Ghost/transparent buttons are reserved for low-emphasis toolbar text.
- Focus is the only state that may draw an accent outline, for keyboard accessibility.
- Do not introduce embossed or beveled dropdown/drop-down controls.
