# IEEE front matter

Read by `paper/build_latex.py --ieee`. Everything else in the IEEE version comes from [`PAPER.md`](../../PAPER.md). IEEE asks for a single-paragraph abstract of at most 250 words, so the abstract is rewritten here from the paper's own; the numbers are the paper's. Authors take one line each, in order: name, affiliation lines, city and e-mail, separated by vertical bars.

## Author

Laksh Saroha | Department of Electronics and Communication Engineering | Thapar Institute of Engineering and Technology | Patiala, India | lsaroha_be24@thapar.edu
Sandeep Mandia | Department of Electronics and Communication Engineering | Thapar Institute of Engineering and Technology | Patiala, India | sandeep.mandia@thapar.edu

## Abstract

Visible cameras on surface vessels degrade in fog, low light, glare and noise, and so do their detectors. Re-estimating a detector's batch-normalization statistics at test time needs no labels, but adapting every frame costs up to 0.023 average precision (AP) at night. The difficulty is deciding which frames to adapt: a dark visible frame may show a dark scene or a failing camera, and the visible image's health score flags every clean night frame. On the Pohang Canal dataset with simulated corruptions, we compare three answers, each pre-registered. IR-gated batch-normalization adaptation (IG-BN) adapts only when the visible frame looks abnormal and a co-mounted thermal camera reports daylight; it needs no gradients or weight changes and can run causally. On the development set it improved 5 of 60 conditions and worsened none. On a held-out daylight run it raised ship AP under fog (+0.037), low light (+0.019), glare (+0.031, with a glare term) and sensor noise (+0.182). A solar-elevation clock in place of the thermal camera was never worse than IG-BN on development and was better wherever the thermal image was degraded. Shallow adaptation, of the first 31 of 130 normalization layers weighted by each block's drift from the training statistics, needs no day–night signal: it improved 24 development and 3 held-out conditions and worsened none, at a small cost at night. We match each answer to the sensors available. On held-out daylight, adapting every frame did best, so the case for gating rests on night frames (thirteen logged scorings).

## Index Terms

Batch normalization, infrared imaging, maritime object detection, sensor fusion, test-time adaptation, thermal imaging, unmanned surface vehicles, visible–infrared detection.
