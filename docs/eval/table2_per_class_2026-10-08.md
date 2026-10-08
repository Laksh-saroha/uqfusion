# Table 2, per class

Written by `scripts/table2_per_class.py`. Day slice: 1,200 paired validation frames (every run but `pohang01`), local AP50-95, caches in `runs/cache_uqslice/` (the ones Table 2 and Figure 2 use). Each row's macro reproduces Table 2's published `map50_95` to 5e-5; the script refuses to write otherwise.

| stream | arm | frames | macro (Table 2) | ship AP | buoy AP | buoy GT | classes emitted |
|---|---|---:|---:|---:|---:|---:|---|
| VIS | σ head | 1200 | 0.3384 | 0.3782 | 0.2987 | 600 | 0, 1 |
| VIS | MC-Dropout | 1200 | 0.3103 | 0.3475 | 0.2731 | 600 | 0, 1 |
| VIS | ensemble (5) | 1200 | 0.3464 | 0.3877 | 0.3050 | 600 | 0, 1 |
| IR | σ head | 1200 | 0.0707 | 0.1414 | 0.0000 | 596 | 0 |
| IR | MC-Dropout | 1200 | 0.0691 | 0.1382 | 0.0000 | 596 | 0 |
| IR | ensemble (5) | 1200 | 0.0744 | 0.1488 | 0.0000 | 596 | 0 |

**Reading.** The IR detector emits class 0 only, yet the IR day labels carry 596 buoy boxes, so the macro scores buoy at exactly 0 and IR's `map50_95` is half its ship AP on every arm. The arm ordering is the same on ship AP as on the macro, for both streams.
