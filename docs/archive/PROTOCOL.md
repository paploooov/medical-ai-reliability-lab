# Forschungsprotokoll v0.1 – vor Modelltraining

Frage: Steigt bei kontrollierter Verschlechterung von PneumoniaMNIST-Testbildern der Anteil hochkonfidenter Fehlvorhersagen?

Primärer Endpunkt: Anteil aller Bilder, die falsch klassifiziert werden und deren vorhergesagte Klassenwahrscheinlichkeit mindestens 0,90 beträgt. Zusätzlich Fehlerrate innerhalb der hochkonfidenten Teilmenge und deren Größe berichten; leere Teilmengen als nicht definiert kennzeichnen.

Geplante Baseline: zunächst logistische Regression, anschließend eine dokumentierte kleine CNN-Baseline. Keine neue Architektur nötig. Train zum Fitten, Validation für Hyperparameter und gegebenenfalls Temperature Scaling; Test erst nach Festlegung der Pipeline. Keine Auswahl schöner Ergebnisse anhand des Testsatzes.

Störungen auf Intensitäten [0,1]: Gaußrauschen mit Standardabweichungen 0,05/0,10/0,20; Gaußunschärfe mit Sigma 0,5/1/2 Pixel; Kontrastfaktoren 0,75/0,50/0,25 um 0,5. Nach Transformation auf [0,1] begrenzen. Original als Kontrolle. Identische Testfälle paarweise vergleichen; Zufallsseeds und Transformationsreihenfolge speichern. Diese Werte sind ein Ausgangsprotokoll und müssen vor der Testauswertung eingefroren werden.

Sekundär: Accuracy, Sensitivität, Spezifität, Confusion Matrix, ROC-AUC, Brier Score und Kalibrierungsdiagramm. Positive Klasse = pneumonia, Entscheidungsschwelle 0,5. Konfidenz = max(p, 1-p). Mehrere Trainingsseeds und gepaarte Bootstrap-Intervalle planen; Bild-Bootstrap bildet keine unbekannte Patientenkorrelation ab.

Speichern: Git-Commit, Paketversionen, Modellparameter, Seeds, Datenprüfsumme, Konfiguration und Vorhersagen je Bild. Varianten desselben Bildes dürfen niemals über Splits verteilt werden. Keine Aussage über Neuheit ohne Literaturvergleich. Keine klinische Leistungsbehauptung aus diesem Benchmark.

Meilensteine: (1) Daten verstehen, (2) Baseline/Validierung, (3) eingefrorenes Störungsexperiment, (4) Unsicherheit und Bericht, (5) erst danach optionale Streamlit-Demo. Für Bewerbungen zählen nachvollziehbare Methoden, ehrliche Grenzen und reproduzierbare Ergebnisse.

## Pilot freeze: 8 November 2026

The first run uses logistic regression (C selected from 0.01, 0.1, 1 by validation log loss) and a 64-unit ReLU multilayer perceptron (40 epochs, best epoch by validation log loss). No CNN is claimed. All inputs are scaled to [0,1], without augmentation. Seed 42; fixed 0.5 decision threshold and 0.9 confidence threshold. Transformations are applied independently, not cumulatively. The same Gaussian noise direction is scaled across severities. Blur uses reflective boundaries. No calibration fitting in this pilot. Paired image bootstrap: 2000 replicates, percentile 95% interval for the change in high-confidence error proportion. Intervals are conditional on this fitted model, not estimates of training variability or patient-level uncertainty. This pilot uses one training seed; multi-seed confirmation and external validation remain future work. Test findings will not be used to retune this run.
