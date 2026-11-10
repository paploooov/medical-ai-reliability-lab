# Medical AI Reliability Lab

**Explorative Studie, Stand 10. November 2026.** Wie gut erkennt ein medizinischer Bildklassifikator seine eigenen Fehler unter Bildstörungen? Ein beobachteter Mechanismus ist der Kollaps des letzten Merkmalsvektors: Bei `h = 0` bleibt nur der Bias-Logit übrig. Nichtleere Merkmalsvektoren garantieren umgekehrt keine zuverlässigen Vorhersagen.

![Übersicht: Testbilder, Repräsentations-Kollaps und Fehler unter Bildstörungen](reports/figures/overview_figure.png)

Links: derselbe Testfall und Nullvektor-Anteile. Mitte: der Bias-bedingte Konfidenz-Spike. Rechts: Fehler unter akzeptierten Vorhersagen; Spannweiten zeigen Minimum–Maximum über drei Seeds, keine Konfidenzintervalle. [PDF](reports/figures/overview_figure.pdf)

Wir vergleichen kleine CNNs auf PneumoniaMNIST mit und ohne Rauschaugmentation, jeweils mit drei Seeds und gleichem Trainingsbudget. Nach Pixel-Deduplizierung sind Training, Modellwahl und Kalibrierung getrennt; die abschließenden Vergleiche verwenden 262 Kalibrierungsfälle und 621 Testrepräsentanten. Code, Gewichte und Vorhersagen sind versioniert; Tests und Prüfsummen sichern die Nachrechnung ab.

| Versuch 07: Kennzahl | Ohne Augmentation | Mit Rauschaugmentation | Differenz mit vorhandenem 95%-Bootstrap-Intervall |
|---|---:|---:|---:|
| Accuracy, Original | 84,11 % | 84,06 % | −0,05 pp [−1,34; +1,18] |
| Accuracy, Rauschen | 39,99 % | 84,00 % | — |
| Fehler unter akzeptierten Rauschfällen | 57,04 % | 12,58 % | — |
| Akzeptanz unter Rauschen | 88,08 % | 82,18 % | — |
| Primärer Kontrast: Unterschied des Fehleranstiegs Rauschen minus Original unter akzeptierten Fällen | — | — | −46,33 pp [−51,96; −40,78] |

Mittel über drei Seeds; Differenzen: Augmentation minus Kontrolle. Die vorhandenen Intervalle beruhen auf 2.000 gepaarten Wiederziehungen von Pixelgruppen bei festen Modellen und Schwellen, nicht auf unabhängigen Patienten oder Trainingswiederholungen; der Original-Accuracy-Vergleich ist sekundär. „—“ bedeutet: kein entsprechendes Intervall berechnet. [Originalauswertung](reports/augmentation/primary.json) · [Methoden und Hauptabbildung](docs/PAPER_SUMMARY.md)

## Externe Validierung (RSNA, Erwachsene)

Dieselben eingefrorenen 28px-Modelle und Schwellen, ohne Nachtraining, auf 2.000 Bildern der RSNA Pneumonia Detection Challenge (1.000 `Lung Opacity`, 1.000 `Normal`, eine Aufnahme pro Patient-ID). Der Rückgang gegenüber dem internen Test betrifft einen Datensatz- und Populationswechsel (PneumoniaMNIST/Guangzhou, pädiatrisch → RSNA/NIH, überwiegend erwachsen; andere Aufnahme, Vorverarbeitung und Labeldefinitionen). Er ist nicht nachgewiesen allein durch den Wechsel von Kindern zu Erwachsenen.

| Clean, Mittel über 3 Seeds [95%-Bootstrap] | Kontrolle intern | Kontrolle extern | Augmentiert intern | Augmentiert extern |
|---|---:|---:|---:|---:|
| Accuracy | 84,1 % [81,3; 86,7] | 73,5 % [71,8; 75,2] | 84,1 % [81,3; 86,7] | 72,9 % [71,1; 74,7] |
| AUROC | 0,941 [0,918; 0,960] | 0,820 [0,802; 0,838] | 0,926 [0,900; 0,947] | 0,804 [0,785; 0,823] |
| ECE (15 Bins) | 0,109 [0,092; 0,135] | 0,141 [0,129; 0,156] | 0,122 [0,106; 0,150] | 0,133 [0,121; 0,150] |
| Annahmerate | 78,5 % [75,7; 81,3] | 69,8 % [68,0; 71,3] | 82,2 % [79,4; 84,8] | 75,2 % [73,6; 76,7] |
| Fehler unter akzeptierten Fällen | 11,2 % [8,8; 13,8] | 20,6 % [18,7; 22,5] | 13,0 % [10,5; 15,9] | 21,8 % [19,8; 23,7] |

Vorab festgelegter Hauptvergleich (Fehler unter akzeptierten Fällen, extern clean, Augmentation minus Kontrolle): **+1,2 pp [+0,4; +1,9]**. Unter Rauschen bleibt der Vorteil der Augmentation erhalten (−23,1 pp [−26,3; −19,9]).

![Risk-Coverage intern vs. extern](reports/external/risk_coverage.png)

AP-Aufnahmen in der Stichprobe haben 79,5 % Prävalenz, PA-Aufnahmen 22,6 %. Auch die AUROC innerhalb der jeweiligen Aufnahmeart ist bei PA niedriger (Kontrolle 0,725 [0,687; 0,760] gegen 0,818 [0,786; 0,849] bei AP). Das begründet eine Untersuchung möglicher Abhängigkeiten von der Aufnahmeart, beweist aber keinen Aufnahmeart-Shortcut. Bericht, Intervalle und Lizenzrecherche: [reports/external](reports/external/README.md) · [docs/DATA_LICENSE_EXTERNAL.md](docs/DATA_LICENSE_EXTERNAL.md).

**Grenzen:** 28×28 Pixel, synthetisches Rauschen und weitere künstliche Störungen, drei Seeds und zwei unterschiedlich zusammengesetzte Benchmarks. Patientenunabhängigkeit ist nicht nachgewiesen. Daraus folgt kein klinischer Anspruch.

## Reproduzieren

Voraussetzung: Python 3.12, Abhängigkeiten aus `requirements.txt` und CPU-PyTorch 2.5.1. Die drei Befehle prüfen die Daten, reproduzieren die Übersicht aus gespeicherten Vorhersagen und führen die Tests aus; kein neues Training. Für Wiederholungen einen neuen Ausgabeordner wählen.

```bash
python scripts/fetch_data.py
python scripts/make_overview.py --output /tmp/medical-ai-overview
python -m unittest discover -s tests -v
```

[Dokumentation und archivierte Berichte](docs/README.md) · [Roadmap](ROADMAP.md) · [Code: MIT / Daten: CC BY 4.0](docs/LICENSING.md)
