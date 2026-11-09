# Verzeichnis der Untersuchungen

Stand: 9. November 2026. **Der aktuelle Stand ist Nachanalyse 08.** Die Nummern kennzeichnen die Reihenfolge der Untersuchungen, nicht unabhängige klinische Studien. Die englischen Detailberichte enthalten Methoden, Ergebnisse und Grenzen.

## 01 – Erste Modelle und Bildstörungen

**Frage:** Werden Vorhersagen bei schlechteren Bildern ungenauer, obwohl das Modell sicher bleibt?

Logistische Regression und ein kleines neuronales Netz, jeweils auf den offiziellen 28×28-Splits. Originalbilder und neun Störungsbedingungen wurden ausgewertet. Das neuronale Netz erreichte 84,0 % Genauigkeit auf Originalbildern. Bei starkem Kontrastverlust sagte es für jeden Fall „pneumonia“ voraus.

[Bericht](PILOT.md) · [Ergebnisse](../../reports/metrics/results.csv) · [Vorhersagen](../../reports/metrics/predictions.npz) · [Code](../../experiments/train.py)

**Grenze:** Ein Trainingsseed; später wurden Überschneidungen zwischen Training und Validierung gefunden. Der Versuch misst noch nicht direkt die Qualität eines Fehlerwarnsystems.

## 02 – Fehler erkennen und Vorhersagen zurückhalten

**Frage:** Kann die Unsicherheit des bereits trainierten Modells falsche Vorhersagen von richtigen unterscheiden?

Keine neuen Modelle. Die gespeicherten Vorhersagen wurden auf Fehlererkennung und zurückgehaltene Fälle untersucht. Eine auf Validierungsdaten gesetzte Schwelle markierte beim neuronalen Modell 54 von 100 Fehlern auf Originalbildern, zugleich aber 107 richtige Vorhersagen.

[Bericht](ERROR_DETECTION.md) · [Ergebnisse](../../reports/metrics/error_detection.csv) · [Schwellen](../../reports/metrics/error_detection.json) · [Code](../../experiments/error_detection.py)

**Grenze:** Nachanalyse des bereits bekannten Testsatzes. Das Warnsignal übersieht Fehler und verursacht zusätzliche Prüfung korrekter Fälle.

## 03 – Bildmodell mit Faltungsschichten und Kalibrierung

**Frage:** Wie unterscheiden sich mehrere Trainingsläufe, und verbessert Temperature Scaling die Wahrscheinlichkeiten?

CNN mit 105.281 Parametern, drei Seeds, je 25 Epochen. Zehn Bildbedingungen jeweils vor und nach Kalibrierung: 60 Ergebniszeilen. Mittlere Genauigkeit auf Originalbildern: 84,8 %. Die Kalibrierung verbesserte den mittleren Test-Brier-Score nicht.

[Bericht](CNN_STUDY.md) · [Protokoll](CNN_PROTOCOL.md) · [Ergebnisse](../../reports/cnn/results.csv) · [Lernverlauf](../../reports/cnn/learning_curves.csv) · [Modelle und Einstellungen](../../reports/cnn/run.json) · [Code](../../experiments/cnn_study.py)

**Grenze:** Offizielle Splits mit den später gefundenen Pixelüberschneidungen. Dieselbe Validierung wurde für Modellwahl und Kalibrierung genutzt. Drei Seeds sind keine drei unabhängigen Patientengruppen.

## Datenprüfung – Warum sich die Methode geändert hat

Alle 80 Ergebniszeilen aus Versuch 01 und 03 wurden anhand der Vorhersagen nachgerechnet. Die geprüften Kennzahlen stimmen. Die Datenprüfung fand acht unterschiedliche pixelidentische Bilder in Training und Validierung, außerdem Wiederholungen innerhalb der Splits. Die Überschneidungszahlen bestehen auch bei 64×64.

[Prüfbericht](AUDIT.md) · [28×28-Prüfung](../../reports/audit/artifact_checks.json) · [64×64-Prüfung](../../reports/audit/resolution_64.json) · [Bildvergleich](../../reports/audit/resolution_comparison.png)

**Folge:** Alte Ergebnisse bleiben als historische Vorversuche erhalten. Versuch 04 verwendet eine neue, ausdrücklich dokumentierte Behandlung pixelidentischer Gruppen.

## 04 – Auflösung und Fehlererkennung

**Frage:** Verändert höhere Auflösung, wie stark die Fehlerrate unter akzeptierten Vorhersagen bei Unschärfe steigt?

28×28 versus 64×64; identisches adaptives CNN mit 21.249 Parametern; je drei Seeds und 15 Epochen. 4.682 Trainingsfälle, je 262 getrennte Fälle für Modellwahl und Kalibrierung, 621 eindeutige Testrepräsentanten. Fünf Bedingungen pro Modell ergeben 30 Ergebniszeilen.

Bei relativer Unschärfe steigt die mittlere Fehlerrate unter akzeptierten Vorhersagen von 11,0 auf 29,6 % bei 28×28 und von 11,0 auf 19,5 % bei 64×64. Gleichzeitig werden unterschiedlich viele Fälle akzeptiert. Der Bericht enthält die vorab festgelegte primäre Gegenüberstellung und bedingte Unsicherheitsintervalle aus 2.000 gepaarten Bootstrap-Stichproben.

[Bericht](RESOLUTION_STUDY.md) · [Festgelegte Methode](RESOLUTION_PROTOCOL.md) · [Ergebnisse](../../reports/resolution/results.csv) · [Primäre Auswertung](../../reports/resolution/primary_analysis.json) · [Lernkurven](../../reports/resolution/learning_curves.pdf) · [Nachprüfung](../../reports/resolution/verification.json) · [Code](../../experiments/resolution_study.py)

**Grenze:** Weiterhin derselbe bekannte Benchmark, keine unabhängige Patientenprüfung. Anderes Modell und Trainingsbudget als in Versuch 03. Ein Ergebnis bei definierter synthetischer Unschärfe belegt keine allgemeine medizinische Zuverlässigkeit.

## 05 – gleicher akzeptierter Anteil

**Frage:** Erklärt die unterschiedliche Zahl zurückgehaltener Vorhersagen den Auflösungsunterschied?

Keine neuen Modelle. Vergleich bei exakt 80 % akzeptierten Fällen, mit labelunabhängiger Behandlung gleicher Konfidenzen und 2.000 gepaarten Bootstrap-Stichproben. Der Unterschied der Fehleranstiege schrumpft gegenüber Versuch 04 von etwa 10,1 auf 2,5 Prozentpunkte; die Regeln beantworten unterschiedliche Fragen. Alle Bedingungen und konkrete Fehlfälle sind sichtbar, einschließlich vollständiger Normal-Klassifikation unter Rauschen bei zwei 28px-Seeds.

[Bericht](SELECTIVE_REVIEW.md) · [Protokoll](SELECTIVE_REVIEW_PROTOCOL.md) · [180 Ergebniszeilen](../../reports/selective_review/matched_coverage.csv) · [PDF](../../reports/selective_review/equal_coverage.pdf) · [Code](../../experiments/selective_review.py)

**Grenze:** Rückblickende Rangdiagnostik anhand derselben Testfälle, keine neue Validierung und keine einsatzfertige Annahmeschwelle. [Externe Datensätze](EXTERNAL_DATA.md) wurden auf fachliche Eignung und dokumentierte Zugangshürden vorgeprüft; noch keine externen Bilder ausgewertet.

## 06 – Rauschdiagnose und 45-Epochen-Budget

**Frage:** Warum kollabieren Modelle unter Rauschen, und reicht längeres Training als Gegenmaßnahme?

Sechs erneute Trainingsläufe, je 45 Epochen; die ersten 15 stimmen mit Versuch 04 überein. Der Zusammenbruch bleibt bestehen. Bei mehreren Modellen werden die letzten Merkmale null, sodass nur der Ausgabebias wirkt. Eine Zusatzdiagnose untersucht sechs Rauschstärken mit drei Rauschseeds, insgesamt 108 Kombinationen.

[Bericht](BUDGET_STUDY.md) · [Protokoll](BUDGET_PROTOCOL.md) · [Ergebnisse](../../reports/budget/results.csv) · [Nachprüfung](../../reports/budget/verification.json)

**Grenze:** Bekannter Benchmark, feste kleine Architektur, keine klinische oder externe Validierung. Die interne Nullvektor-Diagnose erklärt nicht alle Fehler und ist kein medizinischer Befund.

## 07 – Gezielte Rauschaugmentation

**Frage:** Hilft eine vorab festgelegte Gegenmaßnahme, und welche Nachteile hat sie?

Drei neue 28px-Modelle, identisches 45-Epochen-Budget und identische Daten wie die Kontrolle aus Versuch 06. Während des Trainings bekommt die Hälfte der Bilder zufällig Gaußrauschen. Die Rausch-Accuracy steigt im Mittel von 40,0 auf 84,0%; die Original-Accuracy bleibt etwa gleich. Der Fehleranteil unter akzeptierten Originalfällen steigt jedoch von 11,2 auf 13,0%.

[Bericht](AUGMENTATION_STUDY.md) · [Protokoll](AUGMENTATION_PROTOCOL.md) · [48 Vergleichszeilen](../../reports/augmentation/comparison.csv) · [Nachprüfung](../../reports/augmentation/verification.json)

**Grenze:** Absichtlich passende synthetische Störungsfamilie, keine externe Validierung und keine allgemeine Sicherheitsgarantie. [Zusammenhängender Forschungsüberblick](RESEARCH_SUMMARY.md).

## 08 – Aktuell: Unsicherheit der Annahmeschwelle

**Frage:** Was ändert sich, wenn auch die Kalibrierungsstichprobe variiert?

Keine neuen Modelle. Je 2.000 gepaarte Wiederziehungen mit festen bzw. neu berechneten Schwellen sowie 500 Teilmengen pro Kalibrierungsgröße. Große Konfidenz-Gleichstände führen bei fünf Modellen zu 90–94% tatsächlicher statt ungefähr 80% angestrebter Akzeptanz im Kalibrierungssatz. Bei 54 gemeinsamen Ziehungen ist der Hauptvergleich wegen einer leeren akzeptierten Teilmenge undefiniert.

[Bericht](CALIBRATION_STUDY.md) · [Protokoll](CALIBRATION_PROTOCOL.md) · [Ergebnisse](../../reports/calibration/summary.json) · [Nachprüfung](../../reports/calibration/verification.json)

**Grenze:** Empirische Sensitivitätsbereiche auf einem vorhandenen Pool, keine nominal garantierten klinischen Intervalle. Die übrigen 1.946 Ziehungen dürfen die 54 nicht auswertbaren Fälle nicht verdecken.

## Softwaretests sind etwas anderes als Forschungsergebnisse

Die 29 automatisierten Tests prüfen Rechenregeln und technische Eigenschaften: unter anderem Fehlerraten, leere Teilmengen, Konfidenzgleichstände, Temperaturtransformation, Modellformen und Trennung der Pixelgruppen. Zusätzlich werden die gespeicherten Modelle, Vorhersagen und Schwellen nachgerechnet.

[Tests](../../tests) · [Nachprüfung von Versuch 04](../../experiments/verify_resolution.py)

Bestandene Softwaretests zeigen nicht, dass ein Modell medizinisch richtig oder sicher arbeitet. Sie helfen sicherzustellen, dass die ausgewiesenen Zahlen korrekt berechnet wurden.

## Noch offen

Patientenbezogene externe Daten, klinisch begründete Störungen, größere Prüfung verschiedener Trainingsbudgets und die Bewertung eines tatsächlichen Einsatzszenarios. [Literaturüberblick](LITERATURE.md) und [Projektumfang](PROJECT_SCOPE.md) ordnen den bisherigen Stand ein.


## Versuch 09 – Mechanistische Gleichstandsdiagnose

Sechs unveränderte Modelle aus Versuch 06/07; Nachweis von Null-Merkmalsvektoren an den Kalibrierungsschwellen, deterministisches Tie-Breaking und Nullvektor-Ausschluss. [Bericht](MECHANISTIC_TIES.md) · [Prüfartefakt](../../reports/mechanistic_ties/verification.json). Kein neues Training; frühere Artefakte bleiben erhalten.
