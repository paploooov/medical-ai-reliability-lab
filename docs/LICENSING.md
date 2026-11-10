# Code, Daten und abgeleitete Materialien

Der eigene Quellcode dieses Projekts und seine zugehörige Softwaredokumentation stehen unter der [MIT License](../LICENSE). Bei Weitergabe des Codes sind der Copyright-Vermerk und der vollständige Lizenztext beizubehalten. Abhängigkeiten behalten ihre jeweiligen Lizenzen.

## PneumoniaMNIST

Die MIT-Lizenz des Codes gilt **nicht für die medizinischen Ausgangsdaten**. MedMNIST führt PneumoniaMNIST unter [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Bei Nutzung oder Weitergabe sind die Quelle, die Lizenz und vorgenommene Änderungen anzugeben. Die Roharchive werden nicht im Repository mitverteilt.

Attribution für die hier verwendeten Bilder: **PneumoniaMNIST, MedMNIST v2, Yang et al. (2023), [doi:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8), CC BY 4.0; ursprüngliche Thoraxaufnahmen: Kermany et al. (2018), [doi:10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010).** MedMNIST hat die Bilder zentral zugeschnitten und verkleinert. Bildbeispiele dieses Projekts zeigen diese verkleinerten Bilder; in gekennzeichneten Störbedingungen kommen synthetisches Rauschen, Unschärfe oder Kontraständerungen hinzu.

Offizielle Lizenzangabe: [MedMNIST-Datensatzmetadaten, Eintrag pneumoniamnist](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py). Wer stattdessen Originalarchive oder weitere Datensätze verwendet, muss deren eigene Bedingungen prüfen; die MedMNIST-Angabe lizenziert nicht pauschal alle Quellen neu.

## RSNA Pneumonia Detection Challenge

Prüfstand 10. November 2026: Die RSNA-Daten sind **nicht** von der MIT-Code-Lizenz oder der PneumoniaMNIST-Lizenz CC BY 4.0 erfasst. Quellen: [Originalwettbewerb und Regeln](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules), Shih et al. (2019), [doi:10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041), Ausgangsbilder NIH ChestX-ray14, Wang et al. (2017).

Der verwendete [Hugging-Face-Spiegel](https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset/tree/a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4) ist auf Revision `a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4` fixiert. Die Abfrage einer `README.md` an dieser Revision ergab „Entry not found“. Ein belastbarer Lizenztext des Originaldatensatzes und ein Nachweis der Weitergabeberechtigung des Spiegels liegen in dieser Prüfung nicht vor. Die zuvor behauptete Freigabe für nicht-kommerzielle Forschung und Lehre wird daher nicht als bestätigtes Ergebnis dargestellt. Dies ist keine Feststellung, dass die Nutzung unzulässig ist, sondern eine offene Herkunfts- und Rechteprüfung.

Das Repository verteilt keine RSNA-Rohbilder; es enthält Stichproben-IDs, Metadaten, Prüfsummen und numerische Auswertungen. Vor erneutem Datenbezug oder Weitergabe sind die Originalbedingungen und die Berechtigung der Bezugsquelle zu klären. Prüfsummen sichern Dateiintegrität, keine Lizenzrechte.

## Abbildungen und Gewichte

Bildtafeln mit Datensatzbildern behalten deren Herkunft und CC-BY-Attribution; sie werden nicht durch die Code-Lizenz zu MIT-Daten. Rein numerische Ergebnisdateien und trainierte Modellgewichte sind keine pauschal unter MIT neu lizenzierten Ausgangsdaten. Für Modellgewichte wird hier keine gesonderte Lizenz erteilt; ihre Veröffentlichung ist keine automatische allgemeine Nutzungserlaubnis. Diese Abgrenzung lässt die MIT-Freigabe des eigenen Codes unberührt.

Die Software wird ohne Gewährleistung bereitgestellt. Das Repository dokumentiert Forschung und ist keine freigegebene Anwendung zur Diagnose oder Behandlung.
