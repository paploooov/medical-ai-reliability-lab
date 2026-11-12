# THORAXIS: Aufteilung und Fehlerablehnung

**Änderung A1 · 12. November 2026 · Basis: `598d26c`.** Grund: Gutachter-Review; keine Ergebnisse der geplanten Studie gesehen, keine neuen Läufe gestartet. Verbindliche Details und Quellen: [Konfiguration](../configs/thoraxis_source_shift_protocol.json).

**Frage:** Hängt der Fehler unter angenommenen Fällen von der Aufteilung ab, und fängt die Konfidenzablehnung die falsch positiven Normalbilder ab?

## Versuchsplan

**H1, primär:** Offizielle Aufteilung gegen **fünf** gruppierte Neuaufteilungen derselben 5.371 bereinigten Bilder. Split-Seeds: **20261112 bis 20261116**; jeweils Trainingsseeds **17, 42, 83, 123, 2026**. Das ergibt **30 medizinische Modelle**: fünf offizielle Referenzen einmalig, dazu 25 gemischte Modelle. Jede Gruppe bleibt zusammen; jeder Test enthält 618 Bilder mit 231 Normal- und 387 Pneumoniebildern. Auch die anderen Teilmengen behalten ihre Größen und Klassenanzahlen. Unlösbare Zuordnungen führen zum Stopp, nicht zu Ersatzseeds.

Training und Modellwahl bleiben gleich: ursprünglicher ImageNet-Encoder, drei Nachtrainingsepochen, Auswahl nur auf Modellwahldaten. Eingefrorene Präfix- und ursprüngliche Kopfmerkmale werden einmal gecacht; Zuordnung, Gewichte, Vorverarbeitung und Dateien werden per Hash geprüft. Trainierbare Merkmale werden nicht gemeinsam gecacht; Scaler und Köpfe werden je Lauf neu angepasst.

Primär mitteln wir den Fehlerunterschied bei **80 % Abdeckung**, offiziell minus gemischt, gleichgewichtet über alle 25 Split/Seed-Paare. Wichtigste Nebenkennzahl bleibt **Spezifität bei Schwelle 0,5**. Die feste interne Ablehnungsschwelle wird zusätzlich geprüft; gleiche Konfidenzen an der R80-Grenze zählen anteilig.

**H2:** Konfidenz falsch positiver Normalbilder, Akzeptanz und Fehler nach wahrer Klasse berichten. Im gemischten Test zusätzlich nach ursprünglichem Train/Test getrennt ausweisen, mit Bild-, Gruppen- und Klassenanzahlen; die globale Ablehnungsregel beibehalten.

**H3, explorativ:** Lanczos gegen bilinear bei allen eingefrorenen Modellen; bilineare Merkmale brauchen einen eigenen Cache. **Shortcut-Prüfung:** Herkunfts-CNN nur auf Normalbildern, fünf gruppierte Außenfolds mit innerer Modellwahl. Neu ist die explorative Spearman-Korrelation seines Out-of-fold-Herkunftsscores mit dem Pneumonie-Logit, ausschließlich auf Normalbildern des jeweiligen medizinischen Tests, zusätzlich innerhalb jeder Herkunft. Sie misst, ob beide Bewertungen ähnlich geordnet sind. **Lungenmaskierung ist eine spätere Studie und nicht Teil dieser Etappe.**

## Statistik, Aufwand und Grenzen

Die vollständige 5×5-Kontrasttabelle, Streuung der fünf Splitmittel und Streuung der fünf Seedmittel werden getrennt berichtet. 2.000 Gruppen-Bootstraps verwenden gemeinsame Gewichte für überlappende Fälle; die wiederverwendete Referenz zählt nicht als 25 unabhängige Modelle. Null Fehler sind kein Nullrisiko; leere Mengen bleiben undefiniert.

Geschätztes Kerntraining: **3,0 Stunden**, mit 25 % Reserve **3,7 Stunden**, einschließlich einmaligem Lanczos-Cache. Grundlage ist der frühere CPU-Lauf; fünf Herkunfts-Fits, Testinferenz, Bootstrap, Download und Splitprüfung sind zusätzlich nötig und ungemessen. Ein bilinearer Cache kostet geschätzt weitere elf Minuten.

Bekannte Testbilder, unbestätigte Patientenkennungen, überlappende Splits und gemeinsame Trainingsdaten begrenzen die Aussage. Unterschiede oder Korrelationen beweisen weder eine Klinikursache noch einen Shortcut; die Rechnung ist eine Planung, keine neue Messung.

## Geplanter Beitrag — fünf Sätze

Wir prüfen die Fehlerablehnung unseres nachtrainierten Modells. Fünf Aufteilungen machen die Abhängigkeit von einer einzelnen Mischung sichtbar. Gleiche Trainingsbedingungen ermöglichen einen nachvollziehbaren Vergleich. Herkunftsgruppen und Score-Korrelationen liefern ergänzende Hinweise. Neuheit und klinische Bedeutung müssen weiterhin geprüft werden.

**Geprüfte Literatur:** [Geirhos 2020](https://doi.org/10.1038/s42256-020-00257-z), [DeGrave 2021](https://doi.org/10.1038/s42256-021-00338-7), [Zech 2018](https://doi.org/10.1371/journal.pmed.1002683), [Kermany 2018](https://doi.org/10.1016/j.cell.2018.02.010), [TRIPOD+AI 2024](https://doi.org/10.1136/bmj-2023-078378); Prüfwege unverändert in der Konfiguration. **Protokoll und Manifeste vor Läufen committen; diese Etappe endet ohne Testdatenzugriff oder Experiment.**
