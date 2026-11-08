# Forschungsprotokoll v0.1 – vor Modelltraining

Frage: Steigt bei kontrollierter Verschlechterung von PneumoniaMNIST-Testbildern der Anteil hochkonfidenter Fehlvorhersagen?

Primärer Endpunkt: Anteil aller Bilder, die falsch klassifiziert werden und deren vorhergesagte Klassenwahrscheinlichkeit mindestens 0,90 beträgt. Zusätzlich Fehlerrate innerhalb der hochkonfidenten Teilmenge und deren Größe berichten; leere Teilmengen als nicht definiert kennzeichnen.

Geplante Baseline: zunächst logistische Regression, anschließend eine dokumentierte kleine CNN-Baseline. Keine neue Architektur nötig. Train zum Fitten, Validation für Hyperparameter und gegebenenfalls Temperature Scaling; Test erst nach Festlegung der Pipeline. Keine Auswahl schöner Ergebnisse anhand des Testsatzes.

Störungen auf Intensitäten [0,1]: Gaußrauschen mit Standardabweichungen 0,05/0,10/0,20; Gaußunschärfe mit Sigma 0,5/1/2 Pixel; Kontrastfaktoren 0,75/0,50/0,25 um 0,5. Nach Transformation auf [0,1] begrenzen. Original als Kontrolle. Identische Testfälle paarweise vergleichen; Zufallsseeds und Transformationsreihenfolge speichern. Diese Werte sind ein Ausgangsprotokoll und müssen vor der Testauswertung eingefroren werden.

Sekundär: Accuracy, Sensitivität, Spezifität, Confusion Matrix, ROC-AUC, Brier Score und Kalibrierungsdiagramm. Positive Klasse = pneumonia, Entscheidungsschwelle 0,5. Konfidenz = max(p, 1-p). Mehrere Trainingsseeds und gepaarte Bootstrap-Intervalle planen; Bild-Bootstrap bildet keine unbekannte Patientenkorrelation ab.

Speichern: Git-Commit, Paketversionen, Modellparameter, Seeds, Datenprüfsumme, Konfiguration und Vorhersagen je Bild. Varianten desselben Bildes dürfen niemals über Splits verteilt werden. Keine Aussage über Neuheit ohne Literaturvergleich. Keine klinische Leistungsbehauptung aus diesem Benchmark.

Meilensteine: (1) Daten verstehen, (2) Baseline/Validierung, (3) eingefrorenes Störungsexperiment, (4) Unsicherheit und Bericht, (5) erst danach optionale Streamlit-Demo. Für Bewerbungen zählen nachvollziehbare Methoden, ehrliche Grenzen und reproduzierbare Ergebnisse.
