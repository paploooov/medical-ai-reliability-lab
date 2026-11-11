# Code, Daten und abgeleitete Materialien

Der eigene Quellcode dieses Projekts und seine zugehörige Softwaredokumentation stehen unter der [MIT License](../LICENSE). Bei Weitergabe des Codes sind der Copyright-Vermerk und der vollständige Lizenztext beizubehalten. Abhängigkeiten behalten ihre jeweiligen Lizenzen.

## PneumoniaMNIST

Die MIT-Lizenz des Codes gilt **nicht für die medizinischen Ausgangsdaten**. MedMNIST führt PneumoniaMNIST unter [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Bei Nutzung oder Weitergabe sind die Quelle, die Lizenz und vorgenommene Änderungen anzugeben. Die Roharchive werden nicht im Repository mitverteilt.

Attribution für die hier verwendeten Bilder: **PneumoniaMNIST, MedMNIST v2, Yang et al. (2023), [doi:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8), CC BY 4.0; ursprüngliche Thoraxaufnahmen: Kermany et al. (2018), [doi:10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010).** MedMNIST hat die Bilder zentral zugeschnitten und verkleinert. Bildbeispiele dieses Projekts zeigen diese verkleinerten Bilder; in gekennzeichneten Störbedingungen kommen synthetisches Rauschen, Unschärfe oder Kontraständerungen hinzu.

Offizielle Lizenzangabe: [MedMNIST-Datensatzmetadaten, Eintrag pneumoniamnist](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py). Wer stattdessen Originalarchive oder weitere Datensätze verwendet, muss deren eigene Bedingungen prüfen; die MedMNIST-Angabe lizenziert nicht pauschal alle Quellen neu.

Die ergänzende 224×224-Auswertung verwendet das offizielle `pneumoniamnist_224.npz` aus demselben [MedMNIST-Zenodo-Datensatz](https://zenodo.org/records/10519652), MD5 `d6a3c71de1b945ea11211b03746c1fe1`. Die Auflösungsabbildung zeigt einen fest gewählten Trainingsfall; rechts wurde zusätzlich mit Bicubic auf 28×28 reduziert und wieder auf 224×224 skaliert. Attribution und CC-BY-4.0-Hinweis stehen direkt in der Abbildung. Dies ist keine Freigabe für andere Originalarchive.

Der ergänzende ResNet-18-Encoder verwendet die öffentlichen `IMAGENET1K_V1`-Gewichte von Torchvision. Sie werden nur im ignorierten lokalen Daten-Cache bezogen, mit Herkunft und Prüfsumme dokumentiert und nicht durch dieses Repository unter MIT neu lizenziert oder mitverteilt. Die Projekt-MIT-Lizenz betrifft den eigenen Code, nicht pauschal diese fremden Gewichte oder ImageNet-Daten.

## Original-Thoraxbilder für den 512px-Vergleich

Die neue Studie verwendet **ChestXRay2017.zip** direkt von [Mendeley Data, Kermany, Zhang & Goldbaum (2018), Version 2](https://data.mendeley.com/datasets/rscbjbr9sj/2), DOI **10.17632/rscbjbr9sj.2**. Die offizielle Metadaten-Seite weist **CC BY 4.0** aus. Dateigröße 1.235.512.464 Bytes; die veröffentlichte SHA-256-Prüfsumme `13efc055629733dbab07877f8b3c9f81097840dbcdaa326a8542322c2281ce36` stimmt mit dem geladenen Archiv überein. Die Lizenz wurde für diese konkrete Quelle geprüft und nicht aus MedMNIST abgeleitet.

Änderungen: Grauwertkonvertierung, Duplikatbereinigung, konservative Gruppierung anhand der Dateinamen, Größenänderung mit erhaltenem Seitenverhältnis und schwarzem Padding auf 224/512 Pixel. Eine attributierte Abbildung verwendet einen fest gewählten Trainingsfall. Rohdaten und Feature-Caches bleiben im ignorierten lokalen Datenverzeichnis. Wie in den Quellenbedingungen beschrieben, können gesondert als fremd gekennzeichnete Inhalte zusätzliche Rechte erfordern.

MobileNetV3-Small nutzt die offiziellen Torchvision-Gewichte `IMAGENET1K_V1`; die Referenzencoder verwenden ResNet-18 und ResNet-50 `IMAGENET1K_V1`. Auch diese fremden Gewichte werden nicht pauschal unter der Projekt-MIT-Lizenz neu lizenziert. Lokale PyTorch-/TorchScript-/Core-ML-Artefakte sind Forschungsartefakte, keine veröffentlichte oder klinisch freigegebene Anwendung.

## RSNA Pneumonia Detection Challenge

Prüfstand 10. November 2026: Die RSNA-Daten sind **nicht** von der MIT-Code-Lizenz oder der PneumoniaMNIST-Lizenz CC BY 4.0 erfasst. Quellen: [Originalwettbewerb und Regeln](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules), Shih et al. (2019), [doi:10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041), Ausgangsbilder NIH ChestX-ray14, Wang et al. (2017).

Der verwendete [Hugging-Face-Spiegel](https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset/tree/a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4) ist auf Revision `a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4` fixiert. Die Abfrage einer `README.md` an dieser Revision ergab „Entry not found“. Ein belastbarer Lizenztext des Originaldatensatzes und ein Nachweis der Weitergabeberechtigung des Spiegels liegen in dieser Prüfung nicht vor. Die zuvor behauptete Freigabe für nicht-kommerzielle Forschung und Lehre wird daher nicht als bestätigtes Ergebnis dargestellt. Dies ist keine Feststellung, dass die Nutzung unzulässig ist, sondern eine offene Herkunfts- und Rechteprüfung.

Das Repository verteilt keine RSNA-Rohbilder; es enthält Stichproben-IDs, Metadaten, Prüfsummen und numerische Auswertungen. Vor erneutem Datenbezug oder Weitergabe sind die Originalbedingungen und die Berechtigung der Bezugsquelle zu klären. Prüfsummen sichern Dateiintegrität, keine Lizenzrechte.

## Abbildungen und Gewichte

Bildtafeln mit Datensatzbildern behalten deren Herkunft und CC-BY-Attribution; sie werden nicht durch die Code-Lizenz zu MIT-Daten. Rein numerische Ergebnisdateien und trainierte Modellgewichte sind keine pauschal unter MIT neu lizenzierten Ausgangsdaten. Für Modellgewichte wird hier keine gesonderte Lizenz erteilt; ihre Veröffentlichung ist keine automatische allgemeine Nutzungserlaubnis. Diese Abgrenzung lässt die MIT-Freigabe des eigenen Codes unberührt.

Die Software wird ohne Gewährleistung bereitgestellt. Das Repository dokumentiert Forschung und ist keine freigegebene Anwendung zur Diagnose oder Behandlung.

## Externer ChestMNIST224-Transferbenchmark

Quelle ist das offizielle [MedMNIST-Zenodo-Archiv](https://zenodo.org/records/10519652), Datei `chestmnist_224.npz`. Die [offiziellen Metadaten](https://github.com/MedMNIST/MedMNIST/blob/ca3b04fa55c7e6f9e478c362af1f290a1a5cf3dc/medmnist/info.py) weisen ChestMNIST als **CC BY 4.0** aus; MD5 `45bd33e6f06c3e8cdb481c74a89152aa`, 3.889.293.042 Bytes. Attribution: Yang et al., *MedMNIST v2*, Scientific Data 10, 41 (2023), [doi:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8); Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, CVPR (2017), NIH ChestX-ray14.

Verarbeitung: offizielle 224px-Testbilder, exakte Pixel-Deduplizierung, Ausschluss widersprüchlicher Labelvektoren, festgelegte Stichprobe nach textbasierten Krankheitslabels und Inferenz mit dem unveränderten 224px-Modell. Das vollständige Archiv bleibt lokal im ignorierten Datenverzeichnis. Diese Quelle und Lizenz ersetzen keine noch offene Prüfung des zuvor verwendeten RSNA-Spiegels; beide Auswertungen sind getrennt zu behandeln.

Zusätzlich wurden öffentliche Zugangsseiten für Montgomery und VinDr-CXR geprüft. Montgomery lieferte bei den Folgeabrufen HTML beziehungsweise HTTP 403 statt nutzbarer Daten/Originalbedingungen und wurde nicht ausgewertet. [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) verlangt laut öffentlicher Seite einen credentialed PhysioNet-Zugang und eine unterzeichnete Nutzungsvereinbarung; Daten wurden nicht bezogen. Damit liegt weiterhin keine neue unabhängige klinische 512px-Kohorte vor.
