# THORAXIS – Zuverlässigkeit von Pneumonie-Klassifikatoren auf Thorax-Röntgenbildern

Explorative Forschungsarbeit, keine klinische Anwendung. Untersucht wird nicht, wie hoch ein AUROC werden kann, sondern **wann ein hoher AUROC täuscht**: durch die Herkunft der Daten, durch die Vorverarbeitung und durch Konfidenzwerte, die Fehler nicht zuverlässig abfangen.

## Fragestellung

Der öffentliche Kermany-Datensatz (Kinder-Röntgenbilder, Pneumonie vs. normal) wird oft mit seiner offiziellen Train/Test-Aufteilung benutzt. Wir fragen:

1. Hängt der Fehler eines vortrainierten Modells davon ab, ob man die offizielle Aufteilung oder eine gruppierte Neuaufteilung derselben Bilder verwendet? (H1, vorab festgelegt)
2. Fängt eine Konfidenzschwelle die dabei entstehenden Fehlalarme ab? (H2, vorab festgelegt)
3. Reagieren die Modelle auf eine Änderung des Resize-Verfahrens, und lassen sich die beiden Bildbestände voneinander unterscheiden? (H3 und Herkunftsanalyse, explorativ)

## Daten und Modell

- **Daten:** Kermany, Zhang & Goldbaum (2018), Mendeley Data, [doi:10.17632/rscbjbr9sj.2](https://data.mendeley.com/datasets/rscbjbr9sj/2), CC BY 4.0. Nach Deduplizierung 5.371 Bilder. Gruppiert wird nach Dateinamen; bestätigte Patientenkennungen gibt es im Datensatz nicht.
- **Modell:** MobileNetV3-Small (Torchvision, ImageNet-Vortraining), 512 × 512 Eingabe. Trainiert wurden eine lineare Klassifikationsschicht und die letzten Encoder-Blöcke. „THORAXIS“ ist nur der Projektname für diese Anpassungen ([Modellkarte](docs/THORAXIS_MODEL_CARD.md)).
- **Ablauf:** Modellwahl und Schwellen nur auf getrennten Modellwahl- und Kalibrierungsdaten. Unsicherheit über Gruppen-Bootstrap (2.000 Ziehungen). Protokoll, Splits und alle 30 Checkpoints wurden vor der Testauswertung in Git veröffentlicht ([Protokoll](docs/THORAXIS_STUDIENPROTOKOLL.md)).

## Hauptergebnisse

| Befund | Wert | Details |
|---|---|---|
| Interner Test, offizielle Aufteilung (1 Modell, 618 Bilder) | AUROC **0,967** [0,954–0,978], aber Spezifität nur **54,5 %** bei Sensitivität 99,5 % | [Bericht](reports/original_512/README.md) |
| Fehler bei 80 % Abdeckung, offiziell minus gemischt (30 Modelle, 5 Splits × 5 Seeds) | **+10,7 Prozentpunkte** (11,3 % vs. 0,6 %); Gruppen-Bootstrap 7,3–14,4 | [H1/H2](docs/THORAXIS_H1_H2.md) |
| Spezifität Kalibrierung → Test, offizieller Arm | **93,7 % → 49,0 %** (gemischt: 92,5 % → 92,3 %) | [H1/H2](docs/THORAXIS_H1_H2.md) |
| Konfidenzablehnung im offiziellen Arm | verwirft 68 % der *richtigen* Normalbefunde; unter angenommenen Normalbildern bleiben 43,8 % falsch | [H1/H2](docs/THORAXIS_H1_H2.md) |
| Herkunft (offizielles Train vs. Test) aus Normalbildern vorhersagbar | Out-of-fold-AUROC **0,869** [0,844–0,891] | [Etappe 3](docs/THORAXIS_ETAPPE3.md) |
| Bilinear statt Lanczos, sonst identisch | Fehler bei 80 % Abdeckung **+4,5 pp** [3,0; 6,3], Spezifität −10,2 pp (offizieller Arm) | [Etappe 3](docs/THORAXIS_ETAPPE3.md) |

**Lesart:** Der hohe AUROC verdeckt eine deutliche Übertragungslücke zwischen den beiden offiziellen Bildbeständen. Die Modelle sind überkonfident (Testgenauigkeit 82,7 % bei mittlerer Konfidenz 95,8 %), und gerade die falsch positiven Normalbilder werden durch die Konfidenzschwelle nicht verlässlich zurückgehalten. Das partielle Nachtraining bringt gegenüber dem reinen linearen Kopf keinen messbaren AUROC-Gewinn (+0,0004 [−0,0008; 0,0015]), wohl aber bessere NLL und Accuracy.

## Was nicht gezeigt ist

- **Keine klinische Validierung** und keine Diagnose-Aussage. Alle Testbilder stammen aus einem öffentlichen Datensatz und waren aus früheren Teilstudien bereits bekannt.
- **Kein kausaler Shortcut-Nachweis.** Herkunft und Trainingszusammensetzung ändern sich gemeinsam; Alters- und Gerätedaten fehlen im Datensatz.
- **Patientenunabhängigkeit nicht nachgewiesen**, da nur Dateinamen-Gruppen vorliegen.
- **Kein externer Test des Hauptmodells.** Ein eingefrorenes 224px-Modell erreichte auf NIH/ChestMNIST nur AUROC 0,673 gegen textbasiertes „No finding“ ([Bericht](reports/external_chest224/README.md)). Eine Auswertung auf VinDr-PCXR ist [vorab protokolliert](docs/THORAXIS_VINDR_PROTOKOLL.md), aber noch nicht durchgeführt.
- Die Vorab-Festlegung erfolgte über öffentliche Git-Commits, nicht über ein externes Register.

## Reproduzieren und prüfen

```bash
make test             # 133 Unit-Tests, synthetische Daten, ca. 30 s, kein Datendownload
make reproduce-512    # Hauptmodell neu trainieren; lädt das Originalarchiv (ca. 1,2 GB)
python scripts/check_thoraxis_primary.py   # Hauptzahl H1 aus gespeicherten Vorhersagen unabhängig nachrechnen
```

Linux, Python 3.12, nur CPU. Training ohne Testauswertung dauerte 15 Minuten bei 1,2 GiB Peak-RAM ([Anleitung und Grenzen](docs/reproduction-512/README.md)). Mit `EXTRA_ARGS='--evaluate-test'` wird zusätzlich der Test ausgewertet.

## Aufbau

- `docs/` – Protokolle, Ergebnisberichte und Modellkarte; [Übersicht aller Vorstudien](docs/VORARBEITEN.md) (28px-CNNs, Kalibrierung, Augmentation, Auflösung, externer Transfer).
- `experiments/`, `scripts/` – Trainings-, Auswertungs- und Prüfskripte. Viele Dateien sind per SHA-256 in Protokollen eingefroren und werden deshalb nicht umformatiert.
- `reports/` – gespeicherte Vorhersagen, Kennzahlen, Grafiken und Checkpoints.
- `tests/` – Unit-Tests der Auswertungslogik.

## Lizenz, Herkunft, Hinweise

Code: MIT. Daten und fremde Gewichte unterliegen eigenen Bedingungen ([Lizenzen](docs/LICENSING.md), [Zitieren](CITATION.cff)). Die Bildarchive liegen nicht im Repository, nur abgeleitete Vorhersagen, Kennzahlen und Gewichte. Entwicklung und Dokumentation erfolgten mit KI-Assistenz. Kein Medizinprodukt.
