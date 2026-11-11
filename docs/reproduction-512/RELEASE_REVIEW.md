# THORAXIS 512-R1 — lokale Veröffentlichungsvorbereitung

Prüfstand: 11. November 2026. Kein Checkpoint-Upload in dieser Etappe.

## Nachgewiesene Quellenbedingungen

- **Trainingsdaten:** Kermany, Zhang & Goldbaum (2018), DOI 10.17632/rscbjbr9sj.2. Die offizielle Mendeley-Seite für Version 2 weist CC BY 4.0 aus; erneut abgerufen. Attribution, Quellenlink und Angaben zu Änderungen sind erforderlich. Gesondert gekennzeichnete Drittinhalte sind laut Quellenseite ausgenommen. Das Paket enthält keine Rohbilder.
- **Torchvision-Code:** Version 0.20.1, BSD 3-Clause. Der vollständige Copyright-/Lizenzhinweis wird beigelegt. Die Projekt-MIT-Lizenz ersetzt diese Bedingungen nicht.
- **Vortrainierte Gewichte:** Torchvision erklärt unter „Pre-trained Model License“ ausdrücklich, dass vortrainierte Modelle eigene, aus dem Trainingsdatensatz abgeleitete Bedingungen haben können. Die BSD-Code-Lizenz ist daher keine hinreichende pauschale Freigabe der ImageNet-Gewichte oder des daraus angepassten Checkpoints. Für die konkrete Weitergabe des MobileNetV3-Small-ImageNet-V1-Derivats liegt in dieser Prüfung keine abschließende gesonderte Freigabe vor.

Quellen:

- https://data.mendeley.com/datasets/rscbjbr9sj/2
- https://raw.githubusercontent.com/pytorch/vision/v0.20.1/LICENSE
- https://github.com/pytorch/vision/blob/v0.20.1/README.md#pre-trained-model-license
- https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.mobilenet_v3_small.html

Wörtlicher relevanter Hinweis aus der Torchvision-README:

> The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case.

Die offizielle MobileNetV3-Small-Dokumentation bestätigt `IMAGENET1K_V1` als die hier verwendete Gewichtsvariante. Der ergänzende Abruf von `https://www.image-net.org/download.php` lieferte HTTP 503; daraus wurden keine Lizenzbedingungen abgeleitet.

## Status der Freigabe

**Technisch vorbereitet; rechtliche Weitergabebedingungen der Ausgangsgewichte noch offen.** Dem Checkpoint wird hier keine MIT- oder CC-BY-Lizenz erteilt. Die CC-BY-Lizenz des Datensatzes allein klärt nicht sämtliche Rechte am Modell. Vor einem Upload müssen die Bedingungen für dieses konkrete Gewichtederivat und eine passende Modelllizenz dokumentiert werden. Dies ist eine offene Prüfung, keine Feststellung, dass Veröffentlichung generell unzulässig ist.

`CITATION.cff` zitiert die MIT-lizenzierte Forschungssoftware; sein Lizenzfeld gilt nicht als Lizenzierung der Modellgewichte. Kollektive Autorenangabe entsprechend dem bestehenden Repository-Copyright; reale Autorennamen, ORCID und ein Release-DOI wurden nicht erfunden.

## Inhalt des vorbereiteten Pakets

Der bestehende THORAXIS-512-R1-Checkpoint, Modellkarte, CITATION.cff, Torchvision-Lizenz, diese Prüfung, Quellenhinweise und Prüfsummen. Der gemessene Wiederholungslauf wird separat gespeichert und ersetzt den historischen Checkpoint nicht. Die Verpackung liegt ausschließlich im ignorierten lokalen Datenverzeichnis.
