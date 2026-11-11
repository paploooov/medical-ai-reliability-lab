# Ergebnis dieser Reproduzierbarkeits-Etappe

Gemessen: **15.17 Minuten** für den vollständigen Trainingsprozess mit Seed 20261121, einschließlich Quellenprüfung, neuem Gewichtedownload, Bildaufbereitung, Merkmalsberechnung, Kopftraining, drei Nachtrainingsepochen, Modellwahl, Kalibrierung und Exportprüfung. Python-/Paketinstallation und erneuter Download des Originalarchivs sind nicht enthalten.

| Gemessene Stufe | Sekunden |
|---|---:|
| source_verification_and_weights_seconds | 3.22 |
| decode_resize_features_seconds | 571.43 |
| head_fit_seconds | 3.03 |
| three_epoch_training_seconds | 330.28 |
| calibration_export_seconds | 2.03 |

**Peak-RAM (Linux-Prozess-RSS): 1185.7 MiB.** Das ist der Höchststand des Trainingsprozesses einschließlich residenter Memory-Map-Seiten, nicht die gesamte Containerbelegung oder ein GPU-Wert. Ein Torch-/BLAS-Rechenthread, CPU Xeon Platinum 8573C, Containerlimit 2 CPU-Äquivalente und 8 GiB RAM.

Laufartefakte: 459.1 MiB; neue Python-Umgebung: 1.11 GiB; gemeinsam nutzbares Originalarchiv: 1.15 GiB.

## Schätzung für 5 Seeds × 2 Aufteilungen

Bei ähnlichen Bildzahlen, identischer Hardware und serieller Ausführung: **2.53 Stunden** für zehn vollständige Trainingsprozesse mit jeweils neu aufgebauten Merkmalen. Mit 25 % Planungsreserve: **3.16 Stunden**. Die Reserve ist keine statistische Unsicherheitsschätzung. RAM steigt bei serieller Ausführung nicht auf das Zehnfache. Zehn getrennt gespeicherte Laufordner benötigen ungefähr 4.48 GiB, zuzüglich gemeinsamer Umgebung und Archiv.

Nicht enthalten: Einrichtung, neue Split-Audits, externe Datenbeschaffung, Testauswertung/Bootstrap und Berichtserstellung. Geldkosten sind Maschinenstunden × tatsächlicher Tarif; hier ist kein Tarif bekannt. Eine zweite Aufteilung wurde nicht erzeugt. Die Schätzung ist aus einem Lauf extrapoliert und keine Messung von zehn Läufen.

## Erledigt und offen

- Frische virtuelle Umgebung mit neu heruntergeladenen Ausgangsgewichten und vollständig neu berechneten Merkmalen; keine trainierten Gewichte oder Feature-Caches übernommen.
- Ein-Befehl-Einstieg und vollständig gepinnte Umgebung ergänzt. Die gemessene Installation mit direkten Pins wurde anschließend gegen die vollständige Lock-Datei ohne Paketänderungen geprüft.
- Gesamter Trainingslauf abgeschlossen; Export gegen zwei Modellwahlbilder geprüft.
- Testauswertung ist explizit optional und blieb ausgeschaltet. Der Testpfad ist deshalb praktisch noch ungeprüft; historische Bootstrap-Vergleiche werden vom neuen Einstieg nicht reproduziert.
- Roharchiv lokal bereitgestellt und per SHA-256 geprüft; ein vollständiger erneuter Archivdownload wurde nicht gemessen.
- Der publizierte Split wird aus dem versionierten Manifest gelesen; keine neue Aufteilung oder automatische Deduplizierungswiederholung.
- Gewählte Epoche: 1. Reproduzierter Checkpoint separat gespeichert; der bisherige THORAXIS-512-R1-Checkpoint bleibt erhalten.
 - Lokales Veröffentlichungspaket erstellt und geprüft. Gewichte-Lizenzprüfung bleibt offen; kein Upload.
- CITATION.cff schema-validiert. Die Autorenangabe ist kollektiv; DOI und persönliche Autorenangaben bleiben einer späteren bestätigten Veröffentlichung vorbehalten.
- 382 bisherige versionierte Ergebnis-/Modellartefakte unverändert.

[Messdaten](training_run.json) · [Prüfprotokoll und Kostenschätzung](stage_checks.json) · [Anleitung](README.md) · [Lizenzprüfung](RELEASE_REVIEW.md)

Die Etappe endet hier. Keine zusätzlichen Seeds, Aufteilungen oder Testauswertungen wurden gestartet.

## Vergleich mit dem bisherigen Checkpoint

Gleiche Epoche gewählt (1); maximale absolute Gewichtsabweichung 4.4107437e-05, RMS-Abweichung 1.9738361e-07. 225 von 242 Tensoren sind bitgleich. Das ist eine numerisch nahe Wiederholung, keine bitidentische Reproduktion. Die Gewichtsdifferenz ersetzt keinen Leistungsnachweis; Testinferenz wurde nicht ausgeführt. [Parametervergleich](checkpoint_comparison.json).
