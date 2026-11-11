# THORAXIS 512-R1 reproduzieren

[Kurzer Etappenbericht mit Messzahlen](STAGE_REPORT.md)

## Ein Befehl

Voraussetzungen: Linux x86-64, Python 3.12 mit `venv`, GNU Make, Internetzugang für Python-Pakete, Originalarchiv und Torchvision-Gewichte sowie ausreichend freier Speicher. Die Messung dieser Etappe verwendet CPU, einen Rechenthread und eine frische virtuelle Umgebung. Andere Betriebssysteme/GPUs sind nicht verifiziert.

Aus einem frischen Checkout:

```bash
make reproduce-512
```

Der geprüfte Downloadweg verwendet einen identifizierenden User-Agent (`THORAXIS-reproduction/1.0`), weil der Quellserver den Standard-Python-User-Agent hier mit HTTP 403 beantwortet. Status, erwartete Dateigröße und ZIP-Anfang wurden geprüft; der gesamte Archivtransfer wurde nicht wiederholt. [Endpunktprüfung](archive_endpoint_check.json).

Das Ziel installiert CPU-Torch 2.5.1 / Torchvision 0.20.1 und die versionierten Abhängigkeiten in `.venv-512`, lädt das offizielle Datenarchiv, prüft dessen SHA-256, verarbeitet die Original-JPEGs und erzeugt einen neuen Lauf unter `data/reproduce-512`. Existierende Laufordner werden nicht überschrieben. Für eine Wiederholung `RUN_DIR=data/reproduce-512-next` setzen.

Die in Git gespeicherte, bereits auditierte Aufteilung aus `reports/original_512/manifest.json` ist die verbindliche Eingabe. Die ursprüngliche Deduplizierung und Split-Erzeugung werden dabei nicht erneut ausgeführt. Es werden keine früheren Bild-/Feature-Caches oder trainierten Köpfe benötigt. Lizenzbedingungen und Quellen siehe [Freigabeprüfung](RELEASE_REVIEW.md).

Der Ablauf trainiert zunächst den linearen Kopf einschließlich Auswahl von C auf Modellwahldaten, anschließend alle drei Nachtrainingsepochen. Die Wahl zwischen Epoche 0–3 erfolgt auf Modellwahldaten. Die Konfidenzschwelle stammt aus Kalibrierungsdaten. Der exportierte vollständige Encoder wird gegen die gecachte Berechnung auf zwei Modellwahlbildern geprüft.

## Tatsächlich gemessener Lauf

```bash
make reproduce-512 VENV=/tmp/thoraxis-fresh-venv \
  RUN_DIR=data/reproduce-512-stage \
  EXTRA_ARGS='--archive data/original_xray/ChestXRay2017.zip'
```

Die virtuelle Umgebung wurde neu erstellt, die Ausgangsgewichte neu heruntergeladen und sämtliche benötigten Bildmerkmale neu berechnet. Nur das unveränderte Originalarchiv wurde lokal bereitgestellt und per SHA-256 geprüft. Die Downloadstrecke des Originalarchivs wurde in diesem Lauf nicht erneut ausgeführt. Der Betriebssystem-Dateicache wurde nicht geleert; die Messung garantiert daher keinen kalten Datenträgercache. Laufzeit des Trainingsprozesses enthält Quellenprüfung und Gewichtedownload, aber nicht die vorherige Python-Installation; die Stufen werden einzeln erfasst.

Ergebnisse des Laufs: `training_run.json` (Laufzeiten, Linux-Peak-RSS, Versionen, Seed, Modellwahl, Prüfsummen), `THORAXIS-512-R1-reproduced.pt` und `prefix.npy`. Die veröffentlichte Aufteilung bleibt fest; `SEED=...` variiert die Trainingsreihenfolge und erzeugt keine zweite Datenaufteilung.

## Auswertung bewusst getrennt

Standardmäßig werden nur Training, Modellwahl, Kalibrierung und Exportprüfung ausgeführt. Testbilder werden nicht dekodiert, für Merkmalsberechnung verwendet oder ausgewertet. Die historische Testliste ist lediglich Bestandteil des gespeicherten Manifests.

Für eine später ausdrücklich gewünschte Reproduktion einschließlich Test-AUROC/Accuracy unterstützt derselbe Befehl `EXTRA_ARGS='--evaluate-test'`. Dieser Pfad ist in dieser Etappe nur implementiert und statisch geprüft; er wurde gemäß Auftrag **nicht ausgeführt**. Er reproduziert nicht automatisch sämtliche historischen Bootstrap- und Vergleichsarm-Auswertungen. Vollständige Test-Reproduzierbarkeit bleibt damit praktisch ungeprüft.

Die Wiederholung trainiert ausschließlich das Hauptmodell. Vergleichsarme, neue Aufteilungen, weitere Seeds und neue Experimente gehören nicht zu dieser Etappe. Bitidentische Gewichte über unterschiedliche Plattformen/BLAS-Versionen werden nicht zugesichert.

## Checkpoint und Zitation

`python scripts/prepare_512_release.py` erstellt einmalig ein lokales Paket des bestehenden historischen Checkpoints unter `data/release-staging/`. Es lädt nichts hoch. [Dateien und Prüfsummen](release_manifest.json), [offene Lizenzfrage](RELEASE_REVIEW.md).

Die neue [`CITATION.cff`](../../CITATION.cff) wurde mit cffconvert 2.0.0 gegen Schema 1.2.0 validiert. Sie zitiert die Software mit der bestehenden kollektiven Autorenangabe; das MIT-Feld lizenziert keine Modellgewichte. Kein DOI oder Personenname wurde erfunden.
