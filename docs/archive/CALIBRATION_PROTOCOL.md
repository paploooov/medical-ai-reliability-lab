# Versuch 08 – Unsicherheit durch die Kalibrierungsstichprobe

Festgelegt am 9. November 2026 vor der neuen Auswertung. Es werden keine Modelle trainiert und keine Auswahlregeln anhand der Testlabels angepasst. Verwendet werden die drei 28px-Kontrollmodelle aus Versuch 06 und die drei augmentierten Modelle aus Versuch 07.

## Hauptfrage

Bleibt der gemessene Unterschied des Rauschen-minus-Original-Anstiegs der akzeptierten Fehlerrate bestehen, wenn zusätzlich zur Teststichprobe auch die 262 Kalibrierungsfälle variieren?

Berechne zunächst die sechs Konfidenzvektoren auf exakt den gespeicherten Kalibrierungsindizes und reproduziere die dokumentierten Schwellen. Danach 2.000 gepaarte Ziehungen mit Zurücklegen: 262 Kalibrierungsindizes pro Ziehung gemeinsam für alle Modelle/Arme, unabhängig davon 621 Testindizes gemeinsam für alle Modelle, Arme und Bedingungen. Die Schwelle ist weiterhin das lineare 20%-Quantil der jeweiligen gezogenen Kalibrierungskonfidenzen; Gleichstände werden durch `>=` angenommen. Keine Falllabels bestimmen die Schwelle.

Zum direkten Vergleich wird auf denselben Testziehungen zusätzlich die frühere feste Schwelle verwendet. Test-RNG 20261110 reproduziert die Stichproben von Versuch 07; Kalibrierungs-RNG 20261111 bleibt davon unabhängig. Hauptkennzahl: Mittel über drei Seeds des Unterschieds der Risikoanstiege, Augmentation minus Kontrolle. Begleitend: ursprüngliche und verrauschte Akzeptanz/Risiken je Arm, Schwellenverteilung und leere akzeptierte Teilmengen. Ziehungen mit leerem Nenner als undefiniert markieren, zählen und bei der betreffenden Intervallbildung auslassen.

## Stichprobengrößen als Zusatzdiagnose

Je 500 zufällige Teilmengen ohne Zurücklegen aus den vorhandenen Kalibrierungsfällen, Größen 32/64/128/262, Seed 20261112. Dieselben ausgewählten Fälle für alle Modelle. Auswertung jeweils auf allen vorhandenen Testfällen. Bei Größe 262 entsteht stets die ursprüngliche Schwelle. Diese Analyse misst Empfindlichkeit innerhalb eines vorhandenen Pools, nicht die Leistung eines künftigen unabhängig rekrutierten Kalibrierungskollektivs. Sie verändert keine Modellgewichte.

## Interpretation und Grenzen

Die empirischen Perzentilbereiche bleiben bedingt auf sechs feste Modelle, bekannte Bilder und Pixelgruppen statt Patienten. Eine zusätzliche Kalibrierungsziehung bildet keine Modellwahl-, Trainings- oder Standortunsicherheit ab. Konfidenzen können große Punktmassen haben, insbesondere bei Bias-only-Ausgaben. An solchen diskreten Schwellen ist eine gewöhnliche Bootstrap-Deutung als nominal verlässliches 95%-Konfidenzintervall problematisch; Bereiche daher als empirische Sensitivitätsbereiche berichten. Eine Zielakzeptanz von 80% auf Kalibrierungsdaten garantiert weder exakt 80% dort bei Gleichständen noch 80% unter Verteilungsänderung.

Alle Schwellen, Teilmengen, Bootstrap-Kennzahlen und Kalibrierungsvorhersagen speichern. Keine Behauptung klinischer Sicherheit und keine Garantie über die maximale Fehlerrate aus diesen Bereichen ableiten.
