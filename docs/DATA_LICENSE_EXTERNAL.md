# Externe Testdaten: Lizenz- und Nutzungsangaben

Stand der Recherche: 10. November 2026. Dies ist eine Dokumentation der auffindbaren Originaltexte, keine Rechtsberatung und keine Feststellung, dass die Nutzung oder der Bezugsweg rechtlich gesichert ist. Wo ein Text nicht geprüft werden konnte, ist das kenntlich gemacht.

Die Rohbilder und DICOM-Dateien liegen nur lokal unter `data/external/` und sind durch `.gitignore` von der Weitergabe mit dem Repository ausgeschlossen. Im Repository stehen die Patient-IDs, die Git-Blob-Prüfsummen und die DICOM-Metadaten der Stichprobe (`configs/external_rsna_sample.csv`).

## 1. RSNA Pneumonia Detection Challenge (2018)

Offizielle Projektseite, abgerufen am 10. November 2026:
[https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)

Wörtlich dort unter „Download the datasets“:

> You may access and use the imaging datasets and annotations for the purposes of academic research and education, and other commercial or non-commercial purposes as long as you meet the following attribution requirements linked below.

Dieselbe Seite verweist auf das PDF „Pneumonia Detection Challenge Terms of Use and Attribution“ (Word-Dokument, Erstellungsdatum 23. April 2020), heruntergeladen am 10. November 2026 von:

[https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf?hash=03B32453B8646245DE131339F09282D1&rev=db1621c90e7d4814b070161153cc600b)

Wörtlich im PDF:

> You may access and use these de-identified imaging datasets and annotations (“the data”) for the purposes of academic research and education, and other commercial or non-commercial purposes as long as you agree to abide by the following provisions:
> 1. Not to make any attempt to identify or contact any individual(s) who may be the subjects of the data.
> 2. If you share or re-distribute the data in any form, to meet the attribution requirements described below:

Für das NIH-Chest-X-ray-Material, aus dem die Challenge-Daten stammen, verlangt das PDF bei Weitergabe: Link zur NIH-Downloadseite [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC); Zitat der CVPR-2017-Arbeit von Wang et al. (ChestX-ray8); sowie die Anerkennung, dass das NIH Clinical Center der Datenanbieter ist.

Für die Bild- und Annotationsdateien der RSNA-STR-Challenge verlangt das PDF bei Weitergabe: Link zur Downloadseite (im PDF angegeben als `https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018`; die heute erreichbare Seite nutzt den oben genannten Pfad unter `/artificial-intelligence/…`) und das Zitat:

George Shih et al., „Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia“, *Radiology: Artificial Intelligence*, 30. Januar 2019, [doi:10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041).

Die frühere Kurzangabe in `reports/external/README.md`, die Nutzung sei auf nicht-kommerzielle Forschung und Lehre beschränkt, entspricht diesen Originaltexten nicht: Die RSNA-Texte nennen ausdrücklich auch kommerzielle Zwecke, gebunden an die genannten Bestimmungen.

## 2. NIH ChestX-ray14 (zugrunde liegende Bilder)

Die RSNA-Challenge verwendet Thoraxaufnahmen aus dem öffentlichen NIH-Datensatz (Wang et al., IEEE CVPR 2017). Die NIH-Downloadseite ist [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC). Beim Abruf am 10. November 2026 war das eine interaktive Box-Oberfläche; ein festes, als Datei lesbares FAQ- oder README-Dokument mit den offiziellen Nutzungsbedingungen konnte von dort nicht als statischer Text gesichert werden. Die originalen NIH-Bedingungen sind deshalb hier nicht wörtlich aus einer NIH-Quelle zitiert.

Google Cloud dokumentiert denselben Datensatz unter [https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) (abgerufen am 10. November 2026) so:

> There are no restrictions on the use of the NIH chest x-ray images. However, the dataset has the following attribution requirements:
>
> - Provide a link to the NIH download site: https://nihcc.app.box.com/v/ChestXray-NIHCC
> - Include a citation to the CVPR 2017 paper: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers, ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471, 2017
> - Acknowledge that the NIH Clinical Center is the data provider

Das ist eine Darstellung von Google, keine von uns geprüfte Primärquelle des NIH Clinical Center. Sie wird hier nur als sekundäre Angabe festgehalten.

## 3. Kaggle-Wettbewerb

Die Challenge lief 2018 auf Kaggle: [https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge). Die Regel-Seite ist [https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules).

Der Wortlaut der Kaggle-Wettbewerbsregeln konnte in dieser Recherche nicht geprüft werden: Die aktuelle Seite antwortete mit einer Begrenzung der Anfragen bzw. lieferte keinen auswertbaren Regeltext, und ein Wayback-Stand vom 9. Juli 2022 ([Archivlink](https://web.archive.org/web/20220709020440/https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules)) ließ sich hier ebenfalls nicht als Text extrahieren. Ob die Kaggle-Regeln zusätzlich zu den RSNA-Bedingungen Beschränkungen enthalten, bleibt daher ungeprüft.

Dieses Projekt hat die Stichprobe laut `reports/external/PROTOCOL.md` nicht über Kaggle, sondern über den unten genannten Hugging-Face-Spiegel bezogen.

## 4. Hugging-Face-Spiegel `Baldezo313/rsna-pneumonia-dataset`

Seiten- und API-Abruf am 10. November 2026, Revision festgehalten wie im Protokoll: `a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4`.

- Datensatzseite: [https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset](https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset). Dort stand „No dataset card yet“. Eine Lizenzangabe war auf der Karte nicht zu finden.
- Hub-API: [https://huggingface.co/api/datasets/Baldezo313/rsna-pneumonia-dataset](https://huggingface.co/api/datasets/Baldezo313/rsna-pneumonia-dataset). Es gab weder ein Feld `license` noch `cardData`. Die `tags` enthielten nur `region:us`. `gated` war `false`.

Die Weitergaberechte dieses inoffiziellen Spiegels sind damit nicht vom Uploader dokumentiert. Ob der Spiegel selbst die RSNA-Attribution bei der Weitergabe erfüllt und ob der Bezug über diesen Spiegel von den Originalbedingungen gedeckt ist, wurde hier nicht festgestellt.

## 5. Angaben in diesem Repository

Für die externe Auswertung werden zitiert: Shih et al. 2019 ([doi:10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041)) und Wang et al., IEEE CVPR 2017; als Bildquelle das NIH Clinical Center über die RSNA-Challenge. Es werden keine Rohbilder mit dem Repository verteilt. Es wird nicht behauptet, dass damit alle Bedingungen erfüllt oder der Spiegelnutzungsweg geklärt sei.
