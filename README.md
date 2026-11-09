<div align="center">

# Medical AI Reliability Lab
### Wie zuverlässig bleibt medizinische KI, wenn sich die Bildqualität verschlechtert?

**Thorax-Röntgenbilder · Fehlererkennung · Kontrollierte Experimente · Reproduzierbare Forschung**

[Aktuelle Ergebnisse](docs/research/CALIBRATION_STUDY.md) · [Alle Untersuchungen](docs/research/EXPERIMENTS.md) · [Forschungsüberblick](docs/research/RESEARCH_SUMMARY.md) · [Modelle](models/README.md)

</div>

## Aktuell: Versuch 08 – Wie stabil ist die Annahmeschwelle?

**9. November 2026 · Nachanalyse ohne neues Training.** Wir variieren jetzt auch die 262 Fälle, anhand derer die Modelle unsichere Vorhersagen zurückhalten sollen.

- Ziel: ungefähr **80 % Akzeptanz** auf den Kalibrierungsfällen.
- Beobachtung: Fünf der sechs Modelle akzeptieren dort tatsächlich **90–94 %**, weil viele Konfidenzwerte genau gleich sind.
- In **54 von 2.000 Wiederziehungen** ist der Hauptvergleich undefiniert: Bei einem Modell bleiben unter Rauschen keine akzeptierten Vorhersagen übrig.
- Unter den auswertbaren Ziehungen bleibt der Vorteil der Rauschaugmentation bestehen. Daraus folgt keine garantiert sichere Fehlerrate.

![Kalibrierungsstichprobe und Schwellenunsicherheit](reports/calibration/bootstrap_sensitivity.png)

[**Vollständige Auswertung und Grenzen →**](docs/research/CALIBRATION_STUDY.md) · [Prüfnachweis](reports/calibration/verification.json) · [PDF](reports/calibration/bootstrap_sensitivity.pdf)

## Versuch 07 – Gezieltes Training gegen Rauschen

**Trainingsvergleich aus Versuch 07.** Nach der Diagnose eines Rauschfehlers vergleichen wir drei 28×28-Modelle mit und ohne Rauschaugmentation. Architektur, Datenaufteilung, Trainingsbudget und Seeds bleiben gleich. Modellwahl und Kalibrierung verwenden getrennte Originaldaten.

| Ergebnis, Mittel über drei Seeds | Ohne Rauschaugmentation | Mit Rauschaugmentation |
|---|---:|---:|
| Accuracy auf Originalbildern | 84,1 % | 84,1 % |
| Accuracy unter definiertem Rauschen | 40,0 % | 84,0 % |
| Fehler unter akzeptierten Rauschfällen | 57,0 % | 12,6 % |
| Akzeptierter Anteil der Rauschfälle | 88,1 % | 82,2 % |
| Fehler unter akzeptierten Originalfällen | 11,2 % | 13,0 % |

**Die Intervention hilft gegen diese Rauschstörung, verbessert aber nicht alles.** Unter den akzeptierten Originalfällen steigt die Fehlerrate leicht. Die Rauschfamilie und primäre Stärke sind absichtlich im Training enthalten; das ist kein Nachweis allgemeiner Robustheit oder klinischer Sicherheit.

![Aktueller Vergleich: Genauigkeit, akzeptierte Fehler und Akzeptanz](reports/augmentation/comparison.png)

Punkte: Mittelwerte der drei Trainingsläufe. Fehlerbalken: Minimum bis Maximum, keine Konfidenzintervalle. Die Bildbedingungen sind getrennte Experimente, keine durchgehende Schweregradskala.

[**Bericht, Unsicherheitsintervalle und Grenzen →**](docs/research/AUGMENTATION_STUDY.md)

[PDF-Abbildung](reports/augmentation/comparison.pdf) · [48 Vergleichszeilen](reports/augmentation/comparison.csv) · [Lernkurven](reports/augmentation/learning_curves.png) · [Nachprüfung](reports/augmentation/verification.json)

## Was wurde davor herausgefunden?

| Versuch | Frage | Ergebnis und Belege |
|---|---|---|
| 01 – Erste Baselines | Bleibt Konfidenz hoch, obwohl Bilder schlechter werden? | Ja, in mehreren Störungsbedingungen. [Vorversuch](docs/research/PILOT.md) |
| 02 – Fehlererkennung | Kann Unsicherheit Fehler markieren? | Nur teilweise; auch richtige Fälle werden zurückgehalten. [Nachanalyse](docs/research/ERROR_DETECTION.md) |
| 03 – CNN und Kalibrierung | Helfen ein Bildmodell und korrigierte Wahrscheinlichkeiten? | Keine allgemeine Verbesserung; deutliche Grenzen. [Bericht](docs/research/CNN_STUDY.md) |
| Datenprüfung | Stimmen Berechnungen und Trennung der Daten? | Kennzahlen bestätigt, Pixelduplikate gefunden; Methode angepasst. [Audit](docs/research/AUDIT.md) |
| 04 – Auflösung | Wie unterscheiden sich 28×28 und 64×64? | Unterschied bei Unschärfe, aber auch andere Akzeptanzraten. [Bericht](docs/research/RESOLUTION_STUDY.md) |
| 05 – Gleiche Akzeptanz | Bleibt der Unterschied bei gleich vielen akzeptierten Fällen? | Er fällt deutlich kleiner aus. [Nachanalyse](docs/research/SELECTIVE_REVIEW.md) |
| 06 – Trainingsbudget | Behebt längeres Training den Rauschfehler? | Nein; der letzte Merkmalsvektor wird bei vielen Fällen null. [Bericht](docs/research/BUDGET_STUDY.md) |
| **07 – Rauschaugmentation** | Hilft gezieltes Training gegen diese Störung? | **Deutliche Verbesserung unter Rauschen, mit Grenzen und Nebenwirkungen.** [Bericht](docs/research/AUGMENTATION_STUDY.md) |
| **08 – Kalibrierungsstichprobe** | Wie stabil sind Annahmeschwellen aus 262 Fällen? | **Aktuell:** Gleichstände und leere Teilmengen begrenzen die Interpretation. [Bericht](docs/research/CALIBRATION_STUDY.md) |

Die Folgeversuche verwenden denselben bereits untersuchten Benchmark. Sie sind keine unabhängigen klinischen Replikationen. Frühere Ergebnisse bleiben nachvollziehbar; verschiedene Architekturen und Datenaufteilungen dürfen nicht als einfache Bestenliste verglichen werden.

## Daten und Aussagekraft

Seit Versuch 04: **4.682 Trainingsfälle**, je **262 getrennte Fälle für Modellwahl und Kalibrierung**, **621 eindeutige Testrepräsentanten**. Gruppen pixelidentischer Bilder werden zusammen behandelt. Unbekannte Patientenüberschneidungen sind damit nicht ausgeschlossen.

Die Untersuchung betrifft öffentliche pädiatrische Röntgenbilder. Synthetisches Rauschen und Unschärfe ersetzen keine externe Prüfung auf anderen Patienten oder Geräten. Höhere MedMNIST-Auflösungen liefern dieselben Fälle mit mehr Bildpunkten. [Externe Datenquellen](docs/research/EXTERNAL_DATA.md) sind vorgeprüft, aber noch nicht ausgewertet.

Die kleinen CNNs haben in den aktuellen Versuchen jeweils **21.249 Parameter**. Eigene trainierte Gewichte, etablierte Architektur – kein medizinisches Foundation-Modell. Die [Forschungsnotiz](docs/research/RESEARCH_SUMMARY.md) erklärt Fragestellung, methodische Änderungen, Befunde und offene Fragen zusammenhängend.

## Reproduzieren

Python 3.12, CPU, keine kostenpflichtigen Dienste. Die aktuelle Nachanalyse nutzt die bereits versionierten Modelle aus Versuch 06 und 07. Für ihre Reproduktion ist kein neues Training erforderlich.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python scripts/inspect_data.py
python experiments/calibration_study.py
python experiments/verify_calibration.py
python -m unittest discover -s tests -v
```

Die Nachanalyse ersetzt nur ihre eigenen Ausgabedateien. Gewichte und Ergebnisse früherer Versuche bleiben erhalten. Rohdaten werden heruntergeladen und geprüft; sie sind nicht in Git enthalten. Die Berichte enthalten die Befehle für frühere Versuche. Das [Colab-Notebook](https://colab.research.google.com/github/paploooov/medical-ai-reliability-lab/blob/main/notebooks/01_data_exploration.ipynb) zeigt bisher nur den Dateneinstieg.

**Prüfstand:** 29 Softwaretests bestanden. Für Versuch 08 wurden 2.000 gepaarte Bootstrap-Ziehungen in beiden Varianten und 2.000 Teilmengen-Auswertungen separat nachgerechnet. Softwaretests sichern Berechnungen ab; sie belegen keine medizinische Sicherheit.

## Orientierung

- `docs/research/`: Berichte, Protokolle, Literatur und Versuchsverzeichnis.
- `configs/`: festgelegte Einstellungen und Datenindizes.
- `experiments/`: Training, Diagnose und Auswertung; `tests/`: Softwareprüfungen.
- `reports/calibration/`: aktuelle Nachanalyse; `reports/augmentation/`, `models/augmentation/`: zugrunde liegende Trainingsstudie.
- Weitere Unterordner in `reports/` und `models/`: erhaltene frühere Versuche.

## Quellen und Nutzung

Daten: [MedMNIST](https://medmnist.com/) / [MedMNIST+](https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md). PneumoniaMNIST wird unter **CC BY 4.0** geführt; gezeigte Bildbeispiele sind daraus abgeleitet. Quelle: Yang et al., *MedMNIST v2*, Scientific Data (2023), [DOI](https://doi.org/10.1038/s41597-022-01721-8), sowie die dort dokumentierte Originaldatenquelle. [Methodischer Literaturüberblick](docs/research/LITERATURE.md).

Forschungsprojekt, keine Anwendung zur Diagnose oder Behandlung. Eine Softwarelizenz wurde noch nicht vergeben; [Daten- und Softwarerechte sind getrennt](docs/LICENSING.md).
