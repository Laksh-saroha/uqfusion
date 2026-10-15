# IEEE front matter

Read by `paper/build_latex.py --ieee`. Everything else in the IEEE version comes from [`PAPER.md`](../../PAPER.md). IEEE asks for a single-paragraph abstract of at most 250 words, so the abstract is rewritten here from the paper's own; the numbers are the paper's. Authors take one line each, in order: name, affiliation lines, city and e-mail, separated by vertical bars.

## Author

Laksh Saroha | Department of Electronics and Communication Engineering | Thapar Institute of Engineering and Technology | Patiala, India | lsaroha_be24@thapar.edu
Sandeep Mandia | Thapar Institute of Engineering and Technology | Patiala, India | sandeep.mandia@thapar.edu

## Abstract

Visible cameras on surface vessels degrade in fog, low light, glare and noise, and so do detectors trained on clean frames. Re-estimating its batch-normalization statistics at test time needs no label, but adapting undegraded frames costs up to 0.031 average precision (AP) at night. The visible camera cannot decide where to adapt: a novelty score of its own image statistics flags every clean night frame as abnormal. We let a co-mounted thermal camera decide. IR-gated batch-normalization adaptation (IG-BN) re-estimates the visible detector's statistics on eight-frame blocks only where the visible frame looks abnormal and a normal-looking thermal frame reads daylight. It takes no gradient, changes no weight and has a causal form. On the Pohang Canal dataset, a canal and port route, every variant was pre-registered before it was scored. IG-BN passed a 60-condition development grid with no condition worse. On one held-out daylight run it replicated its development gains in ship AP50-95 under simulated fog (+0.037), low light (+0.019) and, with a glare term, glare (+0.031); under sensor noise, absent from development, it gained +0.182 in a first test. Inside a ten-pass ensemble the gains add under noise and fog. A registered ablation shows that the day-night separation is needed, since ungated adaptation and a feature-gap gate lost up to 0.023 AP at night, but not the thermal camera: a solar-elevation clock was never worse and was better where the thermal stream was degraded. On the held-out run, adapting every daylight frame did better still.

## Index Terms

Batch normalization, infrared imaging, maritime object detection, sensor fusion, test-time adaptation, thermal imaging, unmanned surface vehicles, visible–infrared detection.
