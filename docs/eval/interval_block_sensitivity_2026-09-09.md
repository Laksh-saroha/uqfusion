# Interval sensitivity to block length

Produced by `scripts/interval_block_sensitivity.py` for R-A3 (`docs/TODO-2026-09-09-architecture-review.md`, finding F04). The resampling unit changes from a single frame to a contiguous block of frames drawn within a run, because consecutive frames of a 10 Hz recording are not independent observations and an iid frame bootstrap therefore reports an interval that is too narrow.

**Headline: this project's intervals are roughly 1.9× too narrow, and that is a lower bound.** At a one-second block (L=10) the standard error is already 1.61× the iid value; at the largest defensible block (L=20, 2s of video) it is 1.95× across pairs, worst 1.99×. The curve is still **rising** there, so the true inflation is larger — this data cannot say how much larger, for the reason below.

**Why the sweep stops at L=20, and why the larger rows must not be quoted.** A moving-block bootstrap needs a block short relative to the series. The shortest run here is **117 frames** (`pohang03`): at L=46 it offers 72 block starts, and at L=200 it offers **one**, so the resample becomes near-deterministic and the variance **collapses**. The median ratio does keep rising to L=100 (3.23×) before falling to 1.47× at L=200 — but everything past L=23 (shortest run / 5) is already unreliable in *either* direction, so the rise there is no more quotable than the fall. The collapse is shown because it is the evidence for the bound, not because it is a result. Honest summary: the inflation is **at least 1.9×**, and this dataset's short runs prevent measuring where it levels off.

## Sensitivity to block length

| comparison | block L | delta | se | 95% CI | CI width | se vs L=1 | spans zero | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sigma − mc | 1 | 0.023557 | 0.002784 | [0.017707, 0.028749] | 0.011042 | 1.00× | **no** | yes |
| sigma − mc | 2 | 0.023557 | 0.002922 | [0.017547, 0.028901] | 0.011354 | 1.05× | **no** | yes |
| sigma − mc | 5 | 0.023557 | 0.003706 | [0.015252, 0.030712] | 0.015460 | 1.33× | **no** | yes |
| sigma − mc | 10 | 0.023557 | 0.004394 | [0.013774, 0.030878] | 0.017103 | 1.58× | **no** | yes |
| sigma − mc | 20 | 0.023557 | 0.005346 | [0.011172, 0.031619] | 0.020448 | 1.92× | **no** | yes |
| sigma − mc | 50 | 0.023557 | 0.008061 | [-0.000082, 0.033877] | 0.033959 | 2.90× | yes | **no — variance collapse** |
| sigma − mc | 100 | 0.023557 | 0.008988 | [-0.006758, 0.028665] | 0.035423 | 3.23× | yes | **no — variance collapse** |
| sigma − mc | 200 | 0.023557 | 0.004208 | [0.011968, 0.028079] | 0.016111 | 1.51× | **no** | **no — variance collapse** |
| sigma − ens | 1 | -0.012337 | 0.001888 | [-0.016338, -0.008886] | 0.007452 | 1.00× | **no** | yes |
| sigma − ens | 2 | -0.012337 | 0.002121 | [-0.016699, -0.008492] | 0.008208 | 1.12× | **no** | yes |
| sigma − ens | 5 | -0.012337 | 0.002631 | [-0.018121, -0.007527] | 0.010594 | 1.39× | **no** | yes |
| sigma − ens | 10 | -0.012337 | 0.003302 | [-0.019326, -0.006575] | 0.012752 | 1.75× | **no** | yes |
| sigma − ens | 20 | -0.012337 | 0.003755 | [-0.021404, -0.006484] | 0.014920 | 1.99× | **no** | yes |
| sigma − ens | 50 | -0.012337 | 0.004083 | [-0.022965, -0.007188] | 0.015777 | 2.16× | **no** | **no — variance collapse** |
| sigma − ens | 100 | -0.012337 | 0.003285 | [-0.023834, -0.010653] | 0.013181 | 1.74× | **no** | **no — variance collapse** |
| sigma − ens | 200 | -0.012337 | 0.002565 | [-0.020795, -0.010913] | 0.009882 | 1.36× | **no** | **no — variance collapse** |
| mc − ens | 1 | -0.035894 | 0.002569 | [-0.040688, -0.030943] | 0.009745 | 1.00× | **no** | yes |
| mc − ens | 2 | -0.035894 | 0.002850 | [-0.041822, -0.030759] | 0.011063 | 1.11× | **no** | yes |
| mc − ens | 5 | -0.035894 | 0.003420 | [-0.042198, -0.029188] | 0.013010 | 1.33× | **no** | yes |
| mc − ens | 10 | -0.035894 | 0.004144 | [-0.043177, -0.026643] | 0.016534 | 1.61× | **no** | yes |
| mc − ens | 20 | -0.035894 | 0.004998 | [-0.044943, -0.024484] | 0.020459 | 1.95× | **no** | yes |
| mc − ens | 50 | -0.035894 | 0.007680 | [-0.047240, -0.013965] | 0.033275 | 2.99× | **no** | **no — variance collapse** |
| mc − ens | 100 | -0.035894 | 0.009606 | [-0.044754, -0.006790] | 0.037964 | 3.74× | **no** | **no — variance collapse** |
| mc − ens | 200 | -0.035894 | 0.003765 | [-0.043536, -0.028995] | 0.014541 | 1.47× | **no** | **no — variance collapse** |

`L=1` is an iid frame bootstrap **within run**, so it isolates block length as the only thing changing. 10 Hz means L=10 is one second of video and L=100 is ten seconds. The **valid** column marks L <= 23 (shortest run 117 / 5); beyond that the variance collapses and the ratio is an artefact.

## The existing global frame bootstrap, for reference

| comparison | delta | se | 95% CI |
|---|---:|---:|---:|
| sigma − mc | 0.023557 | 0.002756 | [0.017748, 0.028695] |
| sigma − ens | -0.012337 | 0.001804 | [-0.016423, -0.009262] |
| mc − ens | -0.035894 | 0.002549 | [-0.040918, -0.030892] |

This is what `apmetrics.bootstrap_delta` produces today. It differs from `L=1` above by also letting run composition vary, so the two are not identical baselines.

## What these intervals cover, and what they cannot

* **Resampling unit:** single frame, drawn within run.
* **Covers:** scene sampling within these recordings, paired across systems.
* **Does not cover:** between-run variation (runs are held fixed by construction).
* **Does not cover:** between-night-run variation — night is entirely pohang01, so this component is not estimable from this data at any block length.
* **Does not cover:** training-seed variation — each cache is one trained model.

Runs present: `{'pohang00': 836, 'pohang01': 1032, 'pohang02': 247, 'pohang03': 117}` (4 runs).

**Night has exactly one run.** No block length, and no number of draws, makes a between-night-run interval estimable from this data. Every night interval this project reports is conditional on `pohang01` and must be stated that way.

## On sign-flip fractions

`blockboot` reports `sign_flip_fraction` and keeps `p_sign_flip` only as a deprecated alias. It is **not a p-value and not the probability that a hypothesis is true** — it is a descriptive property of the resampling distribution under this scheme, and the old name invited exactly the reading R-A3 says to stop making.
