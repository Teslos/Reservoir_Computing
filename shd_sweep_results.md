# SHD leak/speed sweep

1200 train / 800 test, 140 channels × 100 bins, NR=500, seed=1, ridge β=1000.0, sr=1.1. Everything else fixed at RC_SHD.jl values.

## ESN leak (memory depth)

| leak | test acc |
|---|---|
| 0.010 | 0.667 |
| 0.020 | 0.682 |
| 0.030 | 0.675 |
| 0.050 | 0.685 |
| 0.080 | 0.676 |
| 0.120 | 0.669 |
| 0.200 | 0.627 |

**Best leak = 0.05** (acc 0.685)

## FHN speed (slow dynamics = memory depth)

| speed | test acc |
|---|---|
| 0.200 | 0.660 |
| 0.300 | 0.664 |
| 0.400 | 0.642 |
| 0.500 | 0.666 |
| 0.700 | 0.654 |
| 1.000 | 0.621 |

**Best speed = 0.5** (acc 0.666)
