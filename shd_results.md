# SHD (Spiking Heidelberg Digits) reservoir classification

3000 train / 2000 test, 140 channels × 100 bins, 20 classes, NR=500, linear ridge readout. Chance = 0.05.

| Method | Test accuracy |
|---|---|
| raw input | 0.570 |
| discrete ESN | 0.726 |
| FHN oscillators | 0.721 |
