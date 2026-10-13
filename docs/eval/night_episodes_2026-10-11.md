# Night results by episode (descriptive, post hoc)

Not part of any registration. Ship AP, mean over the five Phase 3 systems (one value for `ens5_union`).

## Night test (pohang01 test split), per episode

| episode | frames | shipped | union | lex | VIS | IR | ens5_union | union − shipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test A (11,059–11,575) | 517 | 0.0136 | 0.0406 | 0.0416 | 0.0391 | 0.0136 | 0.0327 | +0.0270 |
| test B (26,161–26,752) | 592 | 0.0506 | 0.1095 | 0.1020 | 0.0956 | 0.0140 | 0.1411 | +0.0589 |

## Night test: the union − shipped interval against block length

Moving blocks of L frames within the run, 1,000 replicates, seed 1. Larger blocks keep more of each episode together; L near an episode's length (517–592) approaches resampling episodes, of which there are two.

| L | 2.5% | 97.5% | width |
|---:|---:|---:|---:|
| 20 | +0.0154 | +0.0400 | 0.0246 |
| 50 | +0.0122 | +0.0476 | 0.0355 |
| 100 | +0.0101 | +0.0549 | 0.0448 |
| 200 | +0.0092 | +0.0589 | 0.0496 |
| 500 | +0.0080 | +0.0441 | 0.0361 |

Observed union − shipped: +0.0273.

## Development night (pohang01 val pairs, draw 941 clean caches), per episode

| episode | shipped | union | lex | VIS | IR | union − shipped |
|---|---:|---:|---:|---:|---:|---:|
| dev A (7,724–8,239) | 0.0127 | 0.1913 | 0.1817 | 0.1773 | 0.0127 | +0.1786 |
| dev B (14,395–14,910) | 0.1457 | 0.4280 | 0.4027 | 0.3540 | 0.1457 | +0.2822 |

`union` is above `shipped` in 4 of 4 episodes. VIS ship AP spans 0.3149 across the 4 episodes.
