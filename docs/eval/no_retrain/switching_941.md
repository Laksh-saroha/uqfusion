# The pooling penalty: switching fusion outputs on random frames (descriptive, draw 941)

`scripts/probe_switch_scale.py`, 2026-10-11, 1:35 AM. Draw 941 is the smoke draw; VIS clean, day frames (1,200), five Phase 3 systems, ship AP, mean over systems. On a random fraction f of frames the output is VIS alone (raw scores); elsewhere it is the union (`crossmodal26m` with the veto off). A random choice carries no information about the frame, so a result below the f-weighted interpolation of the two pure policies is a cost of pooling two score scales into one ranking, not of the choice. `switch_tiered` emits VIS first and IR as the tail on the switched frames.

The union's scores for the same top boxes are 1.13× VIS's raw scores with clean IR (the support bonus) and 0.99× with IR fog s3.

```

== VIS clean, IR clean, day: VIS 0.3485, union 0.3566, union/VIS score ratio (top-5 median) 1.129
frac VIS-only   switch   linear-interp   switch_tiered
0.00           0.3566   0.3566          0.3566
0.10           0.3542   0.3558          0.3543
0.25           0.3502   0.3546          0.3504
0.50           0.3438   0.3525          0.3443
0.75           0.3420   0.3505          0.3429
0.90           0.3435   0.3493          0.3449
1.00           0.3485   0.3485          0.3500

== VIS clean, IR fog_s3, day: VIS 0.3485, union 0.3389, union/VIS score ratio (top-5 median) 0.991
frac VIS-only   switch   linear-interp   switch_tiered
0.00           0.3389   0.3389          0.3389
0.10           0.3393   0.3399          0.3394
0.25           0.3394   0.3413          0.3395
0.50           0.3402   0.3437          0.3405
0.75           0.3427   0.3461          0.3432
0.90           0.3456   0.3475          0.3461
1.00           0.3485   0.3485          0.3491

```

With clean IR, switching three quarters of the frames to VIS alone scores 0.3420: below VIS alone (0.3485), below the union (0.3566), and 0.0085 below the interpolation. The loss peaks between f = 0.5 and 0.75 and is a third as large where the two scales agree (IR fog s3, ratio 0.99). Any frame-level gate that emits two kinds of output pays this cost on top of whatever its decisions are worth: the vetoed frames' scores are on another scale than the merged frames'. Keeping the demoted stream as a tail does not remove it (`switch_tiered`), because the head itself mixes scales.
