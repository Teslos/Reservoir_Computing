# SHD (Spiking Heidelberg Digits) reservoir classification

3000 train / 1000 test, 70 channels × 100 bins, 20 classes, NR=250, linear ridge readout. Chance = 0.05.

| Method | Test accuracy |
|---|---|
| raw input | 0.400 |
| discrete ESN | 0.520 |
| FHN oscillators | 0.546 |
