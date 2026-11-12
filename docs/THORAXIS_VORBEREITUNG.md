# THORAXIS: Vorbereitung nach Protokoll c44117a

Stand: 12. November 2026. Diese Etappe erzeugt fünf Aufteilungen, einen Lanczos-Cache und genau einen technischen Trainingslauf (offiziell, Seed 17). Sie beantwortet noch keine der Forschungsfragen.

## Aufteilungen

Alle fünf Manifeste bestehen die unabhängige Prüfung: Gruppen bleiben zusammen, Bilder und identische Pixel-Hashes überschneiden sich innerhalb einer Aufteilung nicht. Die Größen und Klassenanzahlen entsprechen exakt dem offiziellen Arm: Training 3.842 (1.088 normal), Modellwahl 442 (134 normal), Kalibrierung 469 (126 normal), Test 618 (231 normal). Die übrigen Bilder zeigen Pneumonie.

| Splitseed | Testbilder aus ursprünglichem offiziellen Test |
|---|---:|
| 20261112 | 75 |
| 20261113 | 66 |
| 20261114 | 75 |
| 20261115 | 94 |
| 20261116 | 76 |

Zwei gemischte Testmengen teilen jeweils 78–106 Bilder (12,6–17,2 % von 618; Jaccard 6,7–9,4 %). Insgesamt enthalten sie 2.296 verschiedene Bilder; ein Bild liegt in allen fünf Testmengen. Alle vorgeschriebenen Seeds wurden verwendet. Der Solver HiGHS 1.8.0 fand für jeden Seed ein Optimum; die Prüfung einer alternativen Lösung ergab keinen Gleichstand innerhalb der vorab festgelegten Toleranz.

Daten: [fünf vollständige Manifeste](../configs/thoraxis_splits/), [Prüfungen, SHA-256 und alle paarweisen Überschneidungen](../configs/thoraxis_splits/checks.json), [vorab festgelegte Ausführungsdetails](../configs/thoraxis_preparation.json).

## Cache und technischer Lauf

Der Cache aller 5.371 Bilder benötigte **497,3 s (8,29 min)** und **885,0 MiB Spitzen-RAM** bei einem CPU-Rechenthread. Die Dateien belegen zusammen **515,3 MiB**. Alle sechs vorher festgelegten Entwicklungsbilder bestanden den Direktvergleich; größte absolute Abweichung: **0,00000334**, Toleranz: 0,0002 plus relativer Anteil. [Cache-Hashes und Einzelprüfungen](../configs/thoraxis_lanczos_cache.json).

Genau ein offizieller Lauf mit Seed 17 und drei Epochen ist abgeschlossen: **372,9 s (6,22 min)**, **1.671,4 MiB Spitzen-RAM**, ein CPU-Rechenthread. Davon entfielen 363,4 s auf die drei Epochen. Er verwendete nur die 4.753 Entwicklungsbilder; der Exportvergleich auf Selection-Bildern bestand. **Keine Testvorhersagen.** [Messwerte und Checkpoint-Hash](../reports/original_512/preparation_seed17.json). Cache-Hashes wurden vor dem Lauf in Commit `51d7f28` gesichert.

## Reproduktion und Grenzen

Mit der gesperrten Umgebung aus `requirements-reproduce-512.lock` im Projektverzeichnis: zuerst `python scripts/prepare_thoraxis_splits.py`, dann `python scripts/prepare_thoraxis_cache.py`. Manifeste und Cache-Hashes vor Training committen; anschließend einmal `python scripts/trial_thoraxis_official.py`. Für Cache und Training `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` setzen. Archiv und ursprüngliche Torchvision-Gewichte müssen an den im Skript angegebenen lokalen Pfaden liegen. Vorhandene Cache-/Laufordner führen zum Stopp; nichts wird überschrieben. Prüfung ohne Bilder: `python -m unittest discover -s tests -p test_thoraxis_preparation.py`.

Keine offenen technischen Fehler. Die eigentlichen Studienläufe und Testauswertungen stehen weiterhin aus. Die Aufteilungen sind abhängig. Dateinamen-Gruppen sind keine bestätigten Patientenidentitäten. Ein Merkmalscache ist keine Testvorhersage; der technische Lauf liefert keine neue Aussage über Testleistung oder Kausalität. Cache und Checkpoint bleiben lokal; committet werden Manifeste, Hashes und Messprotokoll.
