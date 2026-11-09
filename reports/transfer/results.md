# Versuch 10 – vollständige Ergebnistabelle

Alle Anteile in Prozent; Nenner der Accuracy, Nullvektoren, Gleichstände und Akzeptanz: jeweils 621. Der Fehleranteil bezieht sich nur auf akzeptierte Fälle. `undef.` bezeichnet eine leere Annahmemenge. Die Originalschwelle wird auf unveränderten Kalibrierungsbildern bestimmt.

| Arm | Seed | Bedingung | Regel | Accuracy | Nullvektor | Schwellen-Gleichstand | Akzeptanz | Fehler unter akzeptierten | Akzeptiert / Fehler |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| control | 17 | clean | standard | 84.22 | 0.00 | 0.00 | 69.40 | 8.12 | 431 / 35 |
| control | 17 | clean | tie_break_reject_zero | 84.22 | 0.00 | 0.00 | 69.40 | 8.12 | 431 / 35 |
| control | 17 | noise | standard | 44.61 | 0.00 | 0.00 | 68.44 | 47.76 | 425 / 203 |
| control | 17 | noise | tie_break_reject_zero | 44.61 | 0.00 | 0.00 | 68.28 | 47.64 | 424 / 202 |
| control | 17 | blur | standard | 71.82 | 0.00 | 0.00 | 73.11 | 17.84 | 454 / 81 |
| control | 17 | blur | tie_break_reject_zero | 71.82 | 0.00 | 0.00 | 72.95 | 17.88 | 453 / 81 |
| control | 17 | contrast | standard | 73.75 | 0.00 | 0.00 | 59.90 | 8.60 | 372 / 32 |
| control | 17 | contrast | tie_break_reject_zero | 73.75 | 0.00 | 0.00 | 59.74 | 8.63 | 371 / 32 |
| control | 42 | clean | standard | 85.02 | 12.08 | 12.08 | 81.96 | 11.79 | 509 / 60 |
| control | 42 | clean | tie_break_reject_zero | 85.02 | 12.08 | 12.08 | 69.89 | 13.82 | 434 / 60 |
| control | 42 | noise | standard | 37.68 | 96.62 | 96.62 | 96.94 | 61.30 | 602 / 369 |
| control | 42 | noise | tie_break_reject_zero | 37.68 | 96.62 | 96.62 | 0.32 | 100.00 | 2 / 2 |
| control | 42 | blur | standard | 66.83 | 0.64 | 0.64 | 89.21 | 29.42 | 554 / 163 |
| control | 42 | blur | tie_break_reject_zero | 66.83 | 0.64 | 0.64 | 88.57 | 29.64 | 550 / 163 |
| control | 42 | contrast | standard | 67.15 | 0.16 | 0.16 | 84.70 | 26.62 | 526 / 140 |
| control | 42 | contrast | tie_break_reject_zero | 67.15 | 0.16 | 0.16 | 84.54 | 26.67 | 525 / 140 |
| control | 83 | clean | standard | 83.09 | 12.24 | 12.24 | 84.22 | 13.58 | 523 / 71 |
| control | 83 | clean | tie_break_reject_zero | 83.09 | 12.24 | 12.24 | 71.98 | 15.88 | 447 / 71 |
| control | 83 | noise | standard | 37.68 | 98.87 | 98.87 | 98.87 | 62.05 | 614 / 381 |
| control | 83 | noise | tie_break_reject_zero | 37.68 | 98.87 | 98.87 | 0.00 | undef. | 0 / 0 |
| control | 83 | blur | standard | 65.06 | 0.64 | 0.64 | 93.40 | 32.59 | 580 / 189 |
| control | 83 | blur | tie_break_reject_zero | 65.06 | 0.64 | 0.64 | 92.75 | 32.81 | 576 / 189 |
| control | 83 | contrast | standard | 64.09 | 0.00 | 0.00 | 90.50 | 31.14 | 562 / 175 |
| control | 83 | contrast | tie_break_reject_zero | 64.09 | 0.00 | 0.00 | 90.50 | 31.14 | 562 / 175 |
| augmentation | 17 | clean | standard | 84.54 | 14.33 | 14.33 | 81.64 | 12.23 | 507 / 62 |
| augmentation | 17 | clean | tie_break_reject_zero | 84.54 | 14.33 | 14.33 | 67.31 | 14.35 | 418 / 60 |
| augmentation | 17 | noise | standard | 84.86 | 13.69 | 13.69 | 81.32 | 11.88 | 505 / 60 |
| augmentation | 17 | noise | tie_break_reject_zero | 84.86 | 13.69 | 13.69 | 67.63 | 13.81 | 420 / 58 |
| augmentation | 17 | blur | standard | 82.29 | 8.21 | 8.21 | 79.87 | 14.92 | 496 / 74 |
| augmentation | 17 | blur | tie_break_reject_zero | 82.29 | 8.21 | 8.21 | 71.66 | 16.18 | 445 / 72 |
| augmentation | 17 | contrast | standard | 74.40 | 1.61 | 1.61 | 79.39 | 20.08 | 493 / 99 |
| augmentation | 17 | contrast | tie_break_reject_zero | 74.40 | 1.61 | 1.61 | 77.78 | 20.50 | 483 / 99 |
| augmentation | 42 | clean | standard | 84.86 | 14.01 | 14.01 | 81.32 | 12.08 | 505 / 61 |
| augmentation | 42 | clean | tie_break_reject_zero | 84.86 | 14.01 | 14.01 | 67.31 | 14.11 | 418 / 59 |
| augmentation | 42 | noise | standard | 84.70 | 14.33 | 14.33 | 81.00 | 11.53 | 503 / 58 |
| augmentation | 42 | noise | tie_break_reject_zero | 84.70 | 14.33 | 14.33 | 66.67 | 13.29 | 414 / 55 |
| augmentation | 42 | blur | standard | 82.13 | 6.60 | 6.60 | 79.23 | 15.24 | 492 / 75 |
| augmentation | 42 | blur | tie_break_reject_zero | 82.13 | 6.60 | 6.60 | 72.62 | 16.63 | 451 / 75 |
| augmentation | 42 | contrast | standard | 72.46 | 0.48 | 0.48 | 80.19 | 22.09 | 498 / 110 |
| augmentation | 42 | contrast | tie_break_reject_zero | 72.46 | 0.48 | 0.48 | 79.71 | 22.22 | 495 / 110 |
| augmentation | 83 | clean | standard | 82.77 | 10.47 | 10.47 | 83.74 | 14.81 | 520 / 77 |
| augmentation | 83 | clean | tie_break_reject_zero | 82.77 | 10.47 | 10.47 | 73.27 | 16.92 | 455 / 77 |
| augmentation | 83 | noise | standard | 82.45 | 11.59 | 11.59 | 84.22 | 14.34 | 523 / 75 |
| augmentation | 83 | noise | tie_break_reject_zero | 82.45 | 11.59 | 11.59 | 72.62 | 16.63 | 451 / 75 |
| augmentation | 83 | blur | standard | 75.20 | 3.86 | 3.86 | 82.77 | 20.43 | 514 / 105 |
| augmentation | 83 | blur | tie_break_reject_zero | 75.20 | 3.86 | 3.86 | 78.90 | 21.43 | 490 / 105 |
| augmentation | 83 | contrast | standard | 66.18 | 0.00 | 0.00 | 90.18 | 30.89 | 560 / 173 |
| augmentation | 83 | contrast | tie_break_reject_zero | 66.18 | 0.00 | 0.00 | 90.18 | 30.89 | 560 / 173 |
