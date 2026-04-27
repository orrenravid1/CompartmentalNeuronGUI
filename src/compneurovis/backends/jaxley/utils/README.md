---
title: Jaxley Utils
summary: Helpers for constructing and positioning Jaxley cells for CompNeuroVis examples and backends.
---

# Jaxley Utils

`compneurovis.backends.jaxley.utils` contains small geometry/layout helpers for Jaxley workflows.

Current helpers:

- `parse_swc(path)`: parse an SWC file into raw node and child maps
- `load_swc_jaxley(path, ...)`: load a single-tree SWC through Jaxley's graph importer
- `load_swc_multi_jaxley(path, ...)`: load every disconnected SWC tree into separate Jaxley cells
- `save_swc_jaxley_cache(path, cache_path, ...)`: cache a single-tree SWC as a solve-ready Jaxley graph payload
- `save_swc_multi_jaxley_cache(path, cache_path, ...)`: cache every disconnected SWC tree for fast reopen
- `load_cached_swc_jaxley(cache_path)`: reopen a cached single-tree SWC
- `load_cached_swc_multi_jaxley(cache_path)`: reopen cached disconnected SWC trees
- `translate_cell_xyzr(cell, offset)`: move a cell's branch `xyzr` coordinates by an `(x, y, z)` offset
- `translate_cells_xyzr(cells, offsets)`: apply offsets across a list of cells
