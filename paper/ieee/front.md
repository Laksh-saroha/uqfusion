# IEEE front matter

Read by `paper/build_latex.py --ieee`. Everything else in the IEEE version comes from [`PAPER.md`](../../PAPER.md). IEEE asks for a single-paragraph abstract of at most 250 words, so the abstract is rewritten here from the paper's own; the numbers are the paper's. Authors take one line each, in order: name, affiliation lines, city and e-mail, separated by vertical bars.

## Author

Laksh Saroha | Department of Electronics and Communication Engineering | Thapar Institute of Engineering and Technology | Patiala, India | lsaroha_be24@thapar.edu
Sandeep Mandia | Department of Electronics and Communication Engineering | Thapar Institute of Engineering and Technology | Patiala, India | sandeep.mandia@thapar.edu

## Abstract

Visible cameras on surface vessels degrade in fog, low light, glare and noise, and so do their detectors. Re-estimating a detector's batch-normalization statistics at test time needs no labels, but doing so on every frame costs about 0.02 average precision (AP) on clean night frames. The difficulty is deciding which frames to adapt: a dark visible frame may show a dark scene or a failing camera, and a health score of the visible image flags every clean night frame as abnormal. We therefore adapt only when the visible frame looks abnormal and an independent day–night signal reports daylight. In IR-gated batch-normalization adaptation (IG-BN), that signal comes from a co-mounted thermal camera; adaptation uses eight-frame blocks, no gradients and no weight changes, and can run causally. On the Pohang Canal dataset with simulated corruptions, every test was pre-registered. On the development set, IG-BN improved 5 of 60 conditions and worsened none. On a held-out daylight run (thirteen logged scorings), it raised ship AP under fog (+0.037), low light (+0.019), glare (+0.031, with a glare term) and sensor noise (+0.182). An ablation showed whole-network adaptation needs the day–night signal: without it, adaptation lost up to 0.023 AP at night. That signal need not come from the thermal camera: a solar-elevation clock from time and position was never worse than the thermal signal on the development set and was better wherever the thermal image was degraded. On held-out daylight frames, adapting every frame did best; the case for gating rests on night frames.

## Index Terms

Batch normalization, infrared imaging, maritime object detection, sensor fusion, test-time adaptation, thermal imaging, unmanned surface vehicles, visible–infrared detection.
