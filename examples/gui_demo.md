# momia3 napari GUI

Interactive dataset browser, pipeline editor and results viewer for
momia3, implemented as an [npe2](https://github.com/napari/npe2) napari
plugin (ARCHITECTURE.md §8). `import momia3` stays napari-free — the
plugin is discovered through the `napari.manifest` entry point and
loaded only when a dock widget is opened.

## Install & launch

```bash
# from the momia3 repo root
uv pip install --python .venv/bin/python -e ".[gui]"
# or: .venv/bin/pip install -e ".[gui]"

.venv/bin/napari
```

In napari, open **Plugins → momia3** and add the three dock widgets
(they cooperate, so open all three):

| Menu entry | Purpose |
|---|---|
| **momia3: Dataset** | Open an `experiment.zarr`, browse FOVs/channels, load Image/Labels layers, manual-edit hook |
| **momia3: Pipeline** | Registry-driven pipeline editor: op dropdown + auto-generated parameter form, step list, JSON export, background run |
| **momia3: Results** | Open a workspace's `cells/edges/tracks.parquet` and render outlines/midlines (Shapes) and lineages (Tracks) |

## Typical workflow

1. **Dataset**: *Open experiment.zarr…* (e.g.
   `examples/demo_workspace/experiment_drift.zarr`). The attribute panel
   shows `pixel_microns`, channels, per-FOV shapes. Pick a channel and
   *Load channel as Image layer* — the `(t, y, x)` stack is wrapped
   lazily in dask (when available) and the time axis maps to napari's
   slider. *Load labels as Labels layer* loads the instance labels
   (editable).
2. **Pipeline**: pick an op from the dropdown (all 45+ registered ops,
   e.g. `io.import_synth`, `segment.threshold`, `measure.cells`,
   `outline.extract`, `midline.extract`, `track.link`). The parameter
   form is generated from the op's pydantic model (Literal→combo box,
   `ge/le`→spin limits, `Optional`/tuple/list→JSON text fields, field
   descriptions→tooltips). *Add step* appends `op(params-summary)` to
   the step list (remove/reorder with the buttons). Required inputs are
   auto-wired to compatible upstream outputs by name/type; set *Input
   store* to feed an existing store into the first store input (a
   `gui.open_store` helper step is prepended). *Export pipeline JSON*
   writes a file equivalent to `examples/pipeline_demo.json`; *Run*
   executes it in a background thread (never blocks the GUI) with a
   progress bar, then auto-loads resulting tables into **Results** and
   the result store's labels into the viewer.
3. **Results**: *Open workspace…* (e.g. `examples/demo_workspace` —
   looks in `tables/` first). Cell outlines (green) and midlines
   (magenta) appear as Shapes layers (subsampled above 5000 cells,
   adjustable), lineages as a Tracks layer built from cells+edges with
   the division graph. *Color tracks by* offers `track_id`, `length_um`,
   and other numeric/pole-age columns when present. Clicking a row in
   the tracks table isolates that track in the Tracks layer; *Show all
   tracks* restores the full view.

## Interactive correction (manual-edit hook)

1. In **Dataset**, load the labels layer, select napari's paint tools
   and fix a segmentation error on the current frame.
2. Click **Recompute cells (current frame)**. The edited frame is
   written back into the store (`set_labels_frame`), and
   `labels_to_cells` → outlines → midlines are re-run for that frame
   only.
3. The **Results** widget (if open) replaces that frame's rows and
   refreshes its layers.

## Headless testing

```bash
# Linux/CI
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui.py -q
# macOS: the offscreen Qt platform has no usable GL context for vispy,
# so tests run on the native platform instead (still never shown);
# this is handled automatically in tests/test_gui.py.
```

The whole module skips cleanly when the `gui` extra is not installed.

## Limitations

- Pipeline auto-wiring connects inputs by same-name-first, then type
  compatibility — ambiguous graphs should be exported to JSON and edited
  by hand (or run via the CLI).
- `Optional` and sequence parameters are edited as JSON text (exact
  round-trip) rather than structured widgets.
- Track isolation shows one track only (its lineage graph edges to
  other tracks are hidden until *Show all tracks*).
- `lineage.refine` interactive correction from the Tracks layer (§8)
  is not yet wired; only the labels-edit → recompute loop is.
