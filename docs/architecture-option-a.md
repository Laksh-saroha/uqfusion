# Option A architecture — `yolo26m` deployed, `yolo12m` as the §7.2 ablation control

Status: **§7.2 decision still open.** The code supports *both* options — the end2end port
is required either way. Option A is the only one that additionally trains `yolo12m`.

Colour key (both diagrams): **green** = implemented and gated on this machine; **amber** =
implemented, training run not yet done (GPU work); **grey** = stock Ultralytics, untouched.

---

## 1. Head anatomy — the only place the two models differ

```mermaid
flowchart TB
    subgraph DEPLOY["yolo26m — the deployed detector (end2end, reg_max=1, no DFL)"]
        direction TB
        N1["neck features P3 / P4 / P5"]
        N1 -->|raw| O2M["one2many branch<br/>cv2 + cv3<br/>training only, discarded at inference"]
        N1 -->|"detached by Ultralytics itself"| O2O["one2one branch<br/>one2one_cv2 + one2one_cv3<br/>4 channels per edge, single distance"]
        N1 -->|"detached by our policy"| CV4A["cv4 sigma-squared branch<br/>4 x log-variance per anchor<br/>0.85M params"]
        O2M --> LOSS["E2ELoss<br/>one2many x 0.8 decaying to 0.1<br/>one2one x remainder<br/>+ beta-NLL on the one2one branch"]
        O2O --> LOSS
        CV4A --> LOSS
        O2O --> PP["postprocess<br/>top-k gather, NO NMS<br/>must be overridden to carry sigma"]
        CV4A --> PP
        PP --> OUT1["boxes + conf + sigma<br/>sigma gathered by the same top-k index"]
    end

    subgraph ABLATE["yolo12m — ablation control only, VIS only (DFL, reg_max=16)"]
        direction TB
        N2["neck features P3 / P4 / P5"]
        N2 -->|raw| BOX["cv2 box branch<br/>64 channels = 4 edges x 16 bins<br/>a 16-bar histogram per edge"]
        N2 -->|raw| CLS["cv3 class branch"]
        N2 -->|"detached by our policy"| CV4B["cv4 sigma-squared branch<br/>identical code to the left<br/>0.85M params"]
        BOX --> LOSS2["GaussianDetectionLoss<br/>box + cls + dfl + beta-NLL"]
        CV4B --> LOSS2
        BOX --> WIDTH["histogram width<br/>per-coordinate std of the 16 bins<br/>free, zero extra parameters"]
        BOX --> NMS["standard NMS"]
        CLS --> NMS
        CV4B --> NMS
        WIDTH --> NMS
        NMS --> OUT2["boxes + conf + sigma + dfl_width<br/>verified: 92 channels at nc=80"]
    end

    classDef build fill:#fde68a,stroke:#b45309,color:#1f2937
    classDef done fill:#bbf7d0,stroke:#15803d,color:#1f2937
    classDef stock fill:#e5e7eb,stroke:#6b7280,color:#1f2937
    class CV4A,LOSS,PP,OUT1 build
    class CV4B,LOSS2,WIDTH,OUT2,BOX,CLS,NMS done
    class N1,N2,O2M,O2O stock
```

The whole §7.2 question is the single node **`histogram width`**: it exists on the right
because `yolo12m` predicts a 16-bar histogram per box edge, and cannot exist on the left
because `yolo26m` predicts one number per edge. Nothing else about the two heads differs —
`cv4` is identical code in both.

---

## 2. Full system — and which model feeds which table

```mermaid
flowchart TB
    VF["Visible frame<br/>2048x1080 native"] --> GV["yolo26m + cv4<br/>VIS weights"]
    IRF["Infrared frame<br/>640x512 native, stays at imgsz 640"] --> GI["yolo26m + cv4<br/>IR weights"]

    GV --> SV["per-box sigma<br/>aleatoric, O2"]
    GV --> MV["Mahalanobis distance<br/>frame-level, O3"]
    GI --> SI["per-box sigma"]
    GI --> MI["Mahalanobis distance"]

    SV --> RV["R_vis = r_frame x r_box"]
    MV --> RV
    SI --> RI["R_ir = r_frame x r_box"]
    MI --> RI

    RV --> FUSE["reliability-weighted WBF<br/>IR to VIS homography, O4"]
    RI --> FUSE
    FUSE --> FINAL["fused detections<br/>+ per-box uncertainty<br/>Table 3"]

    GV --> T2A["Table 2 rows:<br/>Gaussian VIS / IR<br/>vs MC-Dropout vs Deep Ensemble"]
    GI --> T2A

    AB["yolo12m + cv4<br/>VIS only, single seed<br/>NOT part of the fusion system"] --> T2B["Table 2 extra rows:<br/>explicit sigma vs DFL histogram width<br/>both read from ONE set of weights"]

    classDef build fill:#fde68a,stroke:#b45309,color:#1f2937
    classDef done fill:#bbf7d0,stroke:#15803d,color:#1f2937
    classDef stock fill:#e5e7eb,stroke:#6b7280,color:#1f2937
    class GV,GI build
    class AB,T2B done
    class SV,MV,SI,MI,RV,RI,FUSE,FINAL,T2A stock
```

`yolo12m` is a **dead-end branch**: two Table 2 rows and nothing else. Never fused, never
in Table 3, weights ship with no part of the system. Deleting it removes exactly two rows.

---

## 3. What Option A costs over Option B

| | Option B | Option A |
|---|---|---|
| Models trained | `26m` VIS + `26m` IR | + `12m` VIS |
| GPU cost | — | +1 run, ~40–45 h at imgsz 1280 |
| Code kept alive | end2end path only | end2end path **and** the plain-`Detect` path |
| Table 2 | Gaussian / MC / Ensemble | + explicit-vs-DFL pair |
| Risk added | — | none to the deployed system; `12m` path already verified |

`yolo12m` is the right control because it is compute-matched to `yolo26m`, leaving head
design as the only variable:

| | params | GFLOPs | FPS fp32 | Phase 1 mAP50-95 |
|---|---|---|---|---|
| `yolo26m` | 21.9 M | 75.4 | 56.4 | 0.3016 ± 0.0050 |
| `yolo12m` | 20.2 M | 68.1 | 56.1 | 0.2906 ± 0.0046 |

**The claim this buys, stated honestly:** the ablation answers "is the free DFL signal as
good as an explicit variance branch?" *on a DFL backbone*. It does not compare across the
two backbones — that comparison is confounded by the backbone and must not be made.
