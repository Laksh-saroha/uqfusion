# N4c2g (IG-BN-causal+glare) on pohang04's glare cell

`docs/prereg-ttbn-causal-glare-p04-2026-10-14.md` (P-C2G), the twelfth scoring of pohang04. Ship AP against the VIS ground truth, mean over the five systems and the four draws (VIS v2 glare severity 2, draws 941–944, IR clean). Δ = `union_ttbnc2g` − `union`, paired; 95% interval from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1). N4c2g: N4c2's causal BN-mix (previous block's N4 statistics, N = 16, blocks of 8, fp16) under N4g's gate.

**P-C2G: REPLICATES.**

| cell (VIS/IR) | gate | union | Δ vs union [95%] | Δ vs shipped [95%] | N4c2g − N4g [95%] | claim union_ttbnc2g [95%] |
|---|---:|---:|---:|---:|---:|---:|
| glare/clean | 0.810 | 0.1417 | +0.0306 [+0.0287, +0.0325] | +0.0306 [+0.0287, +0.0325] | -0.0003 [-0.0005, -0.0000] | +0.0003 [-0.0012, +0.0023] |

N4c2g − N4g is descriptive (`union_ttbng` is P-G's presort): the price of causality on glare. The claim is fused − max(VIS, IR) with VIS the N4c2g stream.

## glare/clean, descriptive

| arm | ship AP |
|---|---:|
| `union_ttbnc2g/fused` | 0.1724 |
| `union_ttbng/fused` | 0.1726 |
| `union/fused` | 0.1417 |
| `shipped/fused` | 0.1417 |
| `ttbnc2g/vis` | 0.1720 |
| `single/vis` | 0.1381 |
| `single/ir` | 0.0156 |

Gate check: the context gate equalled the frame-statistics gate in all 20 jobs; gated frames per job 10092–10119.

## Against the prediction and the run

* Registered prediction: REPLICATES, Δ about +0.030, slightly below N4g's +0.0309, N4c2g − N4g between −0.0010 and 0.
  Held: Δ +0.0306, N4c2g − N4g −0.0003 [−0.0005, −0.0000], the same size as P-C2's fog and low-light prices
  (−0.0007, −0.0006).
* Built 9:55–11:09 PM (`scripts/ttbn_causal2_glare_p04.py build`, N4c2's sparse passes, fp16): 1,381–1,383 passes per
  draw, 10,092–10,119 gated frames. A first sequential build was stopped after 200 blocks of draw 941 to try two
  processes in parallel; that was slower (1.29 against 1.69 blocks/s combined), so the 943/944 process was stopped
  and those draws rebuilt in one process after 941/942. Every record file was written by one complete run of its
  draw. Scored 11:09–11:14 PM with the verified fast paths (`UQF_FAST_*`); bootstrap of the 40 new presorts
  11:14–11:18 PM (GPU path, checked against the CPU reference on its first presort).
