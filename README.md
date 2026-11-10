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

**Neue Nachanalysen (10. November):** Der Augmentationsvergleich mit **fünf gepaarten Seeds** bestätigt die Richtung des Effekts: −43,7 pp, approximatives 95%-Intervall über Seed-Paare [−56,7; −30,6]. Dieses Intervall beschreibt Trainingsvariation und ist nicht mit dem Bild-Bootstrap der historischen Tabelle gleichzusetzen. [Fünf Seeds und alle Risk-Coverage-Kurven](reports/five_seed/README.md). Temperature Scaling auf internen Kalibrierungsdaten verschlechtert die Test-NLL in allen vier Bedingungen; [Kalibrierungsanalyse](reports/temperature_scaling/README.md). [**Forschungsbericht, 6 Seiten (PDF)**](reports/followup/research_report.pdf) · [Textfassung](reports/followup/research_report.md).

## Externer Test: RSNA

Die eingefrorenen Modelle wurden auf 2.000 externen Thoraxaufnahmen geprüft (1.000 `Lung Opacity`, 1.000 `Normal`; überwiegend Erwachsene). Training und Annahmeschwellen bleiben unverändert. Bei der 28px-Kontrolle sinkt die Accuracy von **84,1 % auf 73,5 %**, während der Fehler unter akzeptierten Fällen von **11,2 % auf 20,6 % [18,7; 22,5]** steigt. Rauschaugmentation hilft extern gegen Rauschen, verbessert den primären Vergleich auf unveränderten Bildern aber nicht: **+1,2 Prozentpunkte akzeptierte Fehler [+0,4; +1,9]** gegenüber der Kontrolle. Intervalle: gepaarter Bild-Bootstrap, bedingt auf feste Modelle und Schwellen.

![Risk-Coverage: interner und externer Test](reports/external/risk_coverage.png)

Mittel und Minimum–Maximum über drei Seeds, keine Konfidenzbänder. [Protokoll, ECE, Reliability Diagrams und Ergebnisse](reports/external/README.md) · [AP/PA-Auswertung](reports/external/README.md#aufnahmeart-appa)

**Grenzen:** 28×28 Pixel (64px als ergänzende Kontrolle), fünf Seeds intern und drei extern, künstliche Störungen und ein wiederholt untersuchter interner Benchmark. Der externe Datensatz- und Populationswechsel umfasst Alter, Aufnahmeart und Labeldefinition; sein Effekt lässt sich nicht allein dem Alter zuordnen. Aufnahmeart und Label sind stark verknüpft. Unabhängige Patienten über verschiedene RSNA-IDs sind nicht nachgewiesen; die Weitergaberechte des verwendeten Datenspiegels sind ungeklärt. Kein klinischer Anspruch.

## Eigenleistung & Werkzeuge

Die neuen Auswertungsskripte, Tests und der Forschungsbericht wurden mit Codex-Unterstützung erstellt und ausgeführt. Die Fortführung und Forschungsausrichtung wurden vom Projektverantwortlichen beauftragt. Frühere persönliche Methodenentscheidungen und Prüfungen sind damit nicht automatisch belegt; eine vollständige individuelle Beitragsbeschreibung bleibt zu ergänzen.

## Reproduzieren

Voraussetzung: Python 3.12, Abhängigkeiten aus `requirements.txt` und CPU-PyTorch 2.5.1. Die drei Befehle prüfen die Daten, reproduzieren die Übersicht aus gespeicherten Vorhersagen und führen die Tests aus; kein neues Training. Für Wiederholungen einen neuen Ausgabeordner wählen.

```bash
python scripts/fetch_data.py
python scripts/make_overview.py --output /tmp/medical-ai-overview
python -m unittest discover -s tests -v
```

[Dokumentation und archivierte Berichte](docs/README.md) · [Roadmap](ROADMAP.md) · [Code- und Datenlizenzen](docs/LICENSING.md)
