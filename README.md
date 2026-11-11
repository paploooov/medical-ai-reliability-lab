# Medical AI Reliability Lab

**Explorative Studie, Stand 11. November 2026.** Wie gut erkennt ein medizinischer Bildklassifikator seine eigenen Fehler unter Bildstörungen? Ein beobachteter Mechanismus ist der Kollaps des letzten Merkmalsvektors: Bei `h = 0` bleibt nur der Bias-Logit übrig. Nichtleere Merkmalsvektoren garantieren umgekehrt keine zuverlässigen Vorhersagen.

![Übersicht: Testbilder, Repräsentations-Kollaps und Fehler unter Bildstörungen](reports/figures/overview_figure.png)

Links: derselbe Testfall und Nullvektor-Anteile. Mitte: der Bias-bedingte Konfidenz-Spike. Rechts: Fehler unter akzeptierten Vorhersagen; Spannweiten zeigen Minimum–Maximum über drei Seeds, keine Konfidenzintervalle. [PDF](reports/figures/overview_figure.pdf)

Wir vergleichen kleine CNNs auf PneumoniaMNIST mit und ohne Rauschaugmentation, inzwischen mit fünf gepaarten Seeds und gleichem Trainingsbudget. Nach Pixel-Deduplizierung sind Training, Modellwahl und Kalibrierung getrennt; die abschließenden Vergleiche verwenden 262 Kalibrierungsfälle und 621 Testrepräsentanten. Code, Gewichte und Vorhersagen sind versioniert; Tests und Prüfsummen sichern die Nachrechnung ab.

| Interner Vergleich, fünf Seeds | Ohne Augmentation | Mit Rauschaugmentation |
|---|---:|---:|
| Accuracy, Original | 83.9% | 84.0% |
| Accuracy, Rauschen | 41.0% | 83.4% |
| Fehler unter akzeptierten Rauschfällen | 54.2% | 11.5% |
| Akzeptanz unter Rauschen | 84.3% | 76.9% |

Primärer Kontrast (Unterschied des Rauschen-minus-Original-Fehleranstiegs unter akzeptierten Fällen, Augmentation minus Kontrolle): **-43.7 pp**. Approximatives 95%-Intervall über fünf Seed-Paare: [-56.7; -30.6] pp. Separater Bild-Bootstrap bei festen Modellen: [-49.0; -38.0] pp. Das sind verschiedene Unsicherheitsquellen; Patientenunabhängigkeit ist nicht belegt. [Alle Seed-Ergebnisse und Risk-Coverage-Kurven](reports/five_seed/README.md) · [Historischer Drei-Seed-Vergleich](reports/augmentation/primary.json).

**Wahrscheinlichkeitskalibrierung:** Temperature Scaling auf internen Kalibrierungsdaten verschlechtert die Test-NLL bei den angepassten Temperaturen in allen vier Bedingungen. Bei zusätzlicher Variation des Temperatur-Fits umfasst der NLL-Differenzbereich der Kontrolle auch null. [Kalibrierungsanalyse](reports/temperature_scaling/README.md). [**Vorheriger 224px-Forschungsbericht, 6 Seiten (PDF)**](reports/followup/research_report.pdf) · [Textfassung](reports/followup/research_report.md).

## Originalbilder: 512px, Nachtraining und größere Modelle

Der neue Ausgangsvergleich nutzt **Original-Thoraxbilder aus der offiziellen Mendeley-Quelle**, mit erhaltenem Seitenverhältnis bei 224 und 512 Pixeln. Nach Entfernen von 32 exakten Duplikaten und 453 Trainingsbildern mit auch im Test vorkommenden Dateinamen-Kennungen verbleiben 3.842 Trainings-, 442 Modellwahl-, 469 Kalibrierungs- und 618 Testbilder. Die Kennungen sind keine unabhängig bestätigten Patientenidentitäten. Diese neue Aufteilung ist nicht direkt mit den früheren MedMNIST-Zahlen vergleichbar.

| Neuer Vergleich | AUROC | Accuracy | Fehler bei 80 % Abdeckung |
|---|---:|---:|---:|
| MobileNet 224, eingefrorener Encoder | 0.9570 | 81.6% | 10.8% |
| MobileNet 512, eingefrorener Encoder | 0.9665 | 80.3% | 11.8% |
| MobileNet 512, partiell nachtrainiert | 0.9669 | 82.7% | 9.2% |
| ResNet-18 224, eingefrorener Encoder | 0.9528 | 79.1% | 12.5% |
| ResNet-50 224, eingefrorener Encoder | 0.9487 | 80.9% | 12.7% |

Der primäre 512-minus-224-AUROC-Unterschied beträgt **+0.0096 [-0.0042; +0.0251]** und ist nicht eindeutig. Partielles Nachtraining senkt dagegen den Fehler bei 80 % Abdeckung um **-2.63 pp [-4.10; -1.19]**; auch die Kontrolle mit gleichem Optimierer und ausschließlich trainiertem Kopf ist schwächer. ResNet-50 mit 23,51 Millionen Parametern liefert keinen eindeutigen AUROC-Gewinn gegenüber ResNet-18.

**Zentrale Grenze:** Beim angepassten 512px-Modell stehen 99,5 % Sensitivität nur 54,5 % Spezifität gegenüber. Eine nachträgliche diagnostische Schwellenanalyse erreicht trotz eines ausschließlich auf Kalibrierungsdaten gesetzten 95%-Spezifitätsziels im Test nur 52,4 %. Mehr Pixel und Parameter lösen diese Übertragungslücke nicht nachweislich. Gruppierte Bootstrap-Intervalle sind auf die ausgewählten Modelle bedingt; neue unabhängige Kliniken fehlen; eine separate Vorverarbeitungsprüfung folgt unten.

[**Neue Forschungsnotiz, 3 Seiten (PDF)**](reports/original_512/research_note.pdf) · [Vollständige Methoden, Kontrollen und Ergebnisse](reports/original_512/README.md).

**SwiftUI-Pfad:** Das nachtrainierte Modell hat 0,93 Millionen Parameter und einen 3,84-MB-PyTorch-Checkpoint. Core ML ist konvertiert; ein synthetisches Referenzsignal und ein [SwiftUI-Tensorprüfer](ios/ResearchPrototype/README.md) sind vorhanden. Xcode-Build, numerische Apple-Laufzeitprüfung und iOS-Bildvorverarbeitung sind noch nicht validiert. Keine fertige Diagnose-App.

**Herkunftsaudit:** Schon Originalbreite und -höhe unterscheiden bei NORMAL-Bildern Entwicklungs- und Testbestand mit AUROC **0,794 [0,760; 0,826]**; mit zusätzlichen Bildmerkmalen **0,849 [0,822; 0,875]**. Bei PNEUMONIA liegt der kombinierte Wert nur bei 0,583. Diese Werte messen Herkunft, keine Krankheitsdiagnose. Gruppierte Out-of-fold-Prüfung; keine kausale Erklärung der Fehler. [Neue Nachanalyse](reports/original_512/source_audit/README.md) · [Kurzbericht (PDF)](reports/original_512/source_audit/research_note.pdf).

**512px-Vorverarbeitung:** Ein zentraler Bildausschnitt verändert **17,5 % [14,3; 20,7]** der Entscheidungen und senkt die Sensitivität von 99,5 % auf 89,4 %. Allein bilineare statt Lanczos-Interpolation verändert 24/618 Entscheidungen. Vier synthetische Bild-/Tensorreferenzen für die spätere App sind erstellt und unter Python geprüft. [Ergebnisse und Referenzen](reports/original_512/preprocessing/README.md).

## Auflösungsprüfung: 224×224

Die bisherigen 28px-Befunde betreffen einen Niedrigauflösungs-Benchmark; sie belegen keine diagnostische Eignung. Ein neuer kontrollierter Vergleich nutzt jetzt das **offizielle 224px-Archiv** und denselben eingefrorenen ResNet-18-Encoder für 224px-Information versus auf 28px reduzierte Information. AUROC: **0.980 vs. 0.960**, gepaarte Differenz **+0.019 [+0.009; +0.032]**. Kein End-to-End-Training und noch kein externer Test dieser neuen Modelle. [Methoden, Bildvergleich und Ergebnisse](reports/high_resolution/README.md).

**Robustheit bei 224px:** Eingangsrauschen erhöht die akzeptierte Fehlerrate des eingefrorenen 224px-Modells von 4,2 % auf 35,6 %, während die Akzeptanz von 73,3 % auf 96,8 % steigt. Der primäre Vergleich des Risikoanstiegs bei gleicher 80%-Abdeckung zeigt keinen eindeutigen Unterschied zwischen den Auflösungen: −0,28 pp [−4,07; +2,21]. [Störungsanalyse mit Kontrolle der Rauschposition](reports/high_resolution/robustness/README.md).

**Gezielte Gegenmaßnahme:** Ein neuer, mit Rauschen trainierter linearer Kopf bei eingefrorenem 224px-Encoder verändert das Risiko bei 80 % Abdeckung unter Testrauschen um **-15.5 pp [-19.3; -11.8]**. Auf clean-Bildern beträgt die Änderung **+0.2 pp [-1.0; +1.0]**. [Rauschaugmentation: Gewinn, Nebenwirkungen und zwei weitere Testrausch-Seeds](reports/high_resolution/augmentation/README.md).

**Störstärke:** Bei doppelter Trainingsrauschstärke (SD 0,2) beträgt die Risikodifferenz bei 80 % Abdeckung **-4.4 pp [-5.8; -2.6]**, Augmentation minus Kontrolle. Das vollständige vorab festgelegte Raster umfasst auch zunehmende Unschärfe und ein gemeinsames Unsicherheitsband. [Grenzen über die Störstärke](reports/high_resolution/severity/README.md).

**Gezielte Blur-Intervention:** Eine Mischung mit 25 % Unschärfetraining verändert das Risiko bei Sigma 4 und 80 % Abdeckung um **-19.0 pp [-22.3; -15.9]** gegenüber dem Rauschkopf. Die Maßnahme wurde nach dem beobachteten Blur-Problem gewählt; clean- und Rauschnachteile sowie eine Kontrolle mit weniger Rauschtraining werden mitberichtet. [Vierarmige Interventionsstudie](reports/high_resolution/mixed_augmentation/README.md).

## Externer Test: RSNA

Die eingefrorenen Modelle wurden auf 2.000 externen Thoraxaufnahmen geprüft (1.000 `Lung Opacity`, 1.000 `Normal`; überwiegend Erwachsene). Training und Annahmeschwellen bleiben unverändert. Bei der 28px-Kontrolle sinkt die Accuracy von **84,1 % auf 73,5 %**, während der Fehler unter akzeptierten Fällen von **11,2 % auf 20,6 % [18,7; 22,5]** steigt. Rauschaugmentation hilft extern gegen Rauschen, verbessert den primären Vergleich auf unveränderten Bildern aber nicht: **+1,2 Prozentpunkte akzeptierte Fehler [+0,4; +1,9]** gegenüber der Kontrolle. Intervalle: gepaarter Bild-Bootstrap, bedingt auf feste Modelle und Schwellen.

![Risk-Coverage: interner und externer Test](reports/external/risk_coverage.png)

Mittel und Minimum–Maximum über drei Seeds, keine Konfidenzbänder. [Protokoll, ECE, Reliability Diagrams und Ergebnisse](reports/external/README.md) · [AP/PA-Auswertung](reports/external/README.md#aufnahmeart-appa)

**Grenzen:** Historische Hauptversuche mit 28×28 Pixeln, externe Kontrolle mit 64px; ergänzender eingefrorener Encoder bei 224px, weiterhin unter Originalauflösung; fünf Seeds intern und drei extern, künstliche Störungen und ein wiederholt untersuchter interner Benchmark. Der externe Datensatz- und Populationswechsel umfasst Alter, Aufnahmeart und Labeldefinition; sein Effekt lässt sich nicht allein dem Alter zuordnen. Aufnahmeart und Label sind stark verknüpft. Unabhängige Patienten über verschiedene RSNA-IDs sind nicht nachgewiesen; die Weitergaberechte des verwendeten Datenspiegels sind ungeklärt. Kein klinischer Anspruch.

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
