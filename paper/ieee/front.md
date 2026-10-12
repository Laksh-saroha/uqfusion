# IEEE front matter

Read by `paper/build_latex.py --ieee`. Everything else in the IEEE version comes from [`PAPER.md`](../../PAPER.md). IEEE asks for a single-paragraph abstract of at most 250 words, so the abstract is rewritten here from the paper's own; the numbers are the paper's.

## Author

Laksh Saroha

## Affiliation

L. Saroha is with the Department of Electronics and Communication Engineering, Thapar Institute of Engineering and Technology, Patiala, India.

## Abstract

Visible cameras at sea degrade in fog, low light, glare and sensor noise, and a detector trained on clean frames degrades with them. Re-estimating its batch-normalization statistics at test time needs no label, but adapting frames that were never shifted costs accuracy: up to 0.031 ship average precision (AP) at night on our development benchmark. A visible camera cannot tell a dark scene from a degraded sensor: its own health score flags every clean night frame as unhealthy. We let a co-mounted thermal camera decide. The method re-estimates the visible detector's batch-normalization statistics on eight-frame blocks only where the visible frame reports itself unhealthy and a healthy infrared frame reads daylight. It takes no gradient, changes no weight, never adapts the infrared detector, and has a causal form. On the Pohang Canal dataset, every variant was pre-registered before it was scored. The method passed development with no cell of 60 worse and replicated on a held-out run: +0.182 [+0.173, +0.192] ship AP under simulated sensor noise, +0.037 under fog, +0.019 under low light and, with a glare term, +0.031 under glare, with no cell worse. Inside a ten-pass ensemble its gains add, giving the best system we measured on every held-out cell with a corrupted visible stream on which it was scored. Earlier uses of the second sensor on the same system, as an uncertainty-weighted voter and as an image-statistic veto, did not survive registration. The corruptions are simulated, and the held-out run is a single daylight run.

## Index Terms

Batch normalization, infrared imaging, maritime object detection, sensor fusion, test-time adaptation, visible–infrared detection.
