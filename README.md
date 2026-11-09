# Medical AI Reliability Lab

**Explorative Studie, Stand 9. November 2026.** Wie gut erkennt ein medizinischer Bildklassifikator seine eigenen Fehler unter Bildstörungen? Ein beobachteter Mechanismus ist der Kollaps des letzten Merkmalsvektors: Bei `h = 0` bleibt nur der Bias-Logit übrig. Nichtleere Merkmalsvektoren garantieren umgekehrt keine zuverlässigen Vorhersagen.

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

**Grenzen:** 28×28 Pixel, synthetisches Rauschen und weitere künstliche Störungen, drei Seeds und ein wiederholt untersuchter Benchmark. Patientenunabhängigkeit ist nicht nachgewiesen; ein externer Test sowie ECE und Reliability Diagrams fehlen. Daraus folgt kein klinischer Anspruch.

## Reproduzieren

Voraussetzung: Python 3.12, Abhängigkeiten aus `requirements.txt` und CPU-PyTorch 2.5.1. Die drei Befehle prüfen die Daten, reproduzieren die Übersicht aus gespeicherten Vorhersagen und führen die Tests aus; kein neues Training. Für Wiederholungen einen neuen Ausgabeordner wählen.

```bash
python scripts/fetch_data.py
python scripts/make_overview.py --output /tmp/medical-ai-overview
python -m unittest discover -s tests -v
```

[Dokumentation und archivierte Berichte](docs/README.md) · [Roadmap](ROADMAP.md) · [Code: MIT / Daten: CC BY 4.0](docs/LICENSING.md)
