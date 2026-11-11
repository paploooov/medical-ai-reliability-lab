# THORAXIS — Modellkarte

THORAXIS ist der Projektname unserer auf Thorax-Röntgenbildern trainierten Forschungsmodelle. Der Hauptstand **THORAXIS 512-R1** basiert auf **MobileNetV3-Small** aus Torchvision mit **ImageNet-1K-V1-Vortraining**. Wir haben eine binäre Klassifikationsschicht auf Pneumonie-/Normalbildern trainiert und anschließend einen Teil des Bildencoders zusammen mit dieser Schicht nachtrainiert. Die Architektur und das ImageNet-Vortraining stammen aus bestehenden Quellen. THORAXIS bezeichnet unsere daraus entstandenen Anpassungen und Gewichte.

Die Namensvergabe führt keine neuen Gewichte oder Trainingsergebnisse ein. Bestehende technische IDs bleiben für die Reproduzierbarkeit gültig.

## Modelle und trainierte Anteile

| Projektname | Technische Ergebnis-ID | Im Projekt trainiert |
|---|---|---|
| **THORAXIS 224-Linear** | `mobile_224_probe` | Lineare Klassifikationsschicht; Bildencoder eingefroren |
| **THORAXIS 512-Linear** | `mobile_512_probe` | Lineare Klassifikationsschicht; Bildencoder eingefroren |
| **THORAXIS 512-R1** | `mobile_512_finetuned` | Klassifikationsschicht und Teil des Bildencoders |

ResNet-18 und ResNet-50 sind Vergleichsarchitekturen und werden unter ihren bisherigen Namen geführt. Die früheren kleinen CNN-Versuche gehören zum Forschungsprojekt, erhalten mit dieser Benennung aber keine neue THORAXIS-Versionsnummer.

## THORAXIS 512-R1: Training und Herkunft

- Ausgangsmodell: Torchvision `mobilenet_v3_small`, Gewichte `IMAGENET1K_V1`.
- Daten: Kermany, Zhang & Goldbaum (2018), Mendeley Data, DOI [10.17632/rscbjbr9sj.2](https://data.mendeley.com/datasets/rscbjbr9sj/2), offizielles `ChestXRay2017.zip`, CC BY 4.0.
- Nach Bereinigung: 3.842 Trainings-, 442 Modellwahl-, 469 Kalibrierungs- und 618 Testbilder. Gruppierung anhand von Dateinamen; unabhängig bestätigte Patientenidentitäten fehlen.
- Eingabe: volle Aufnahme in Graustufen, Lanczos-Skalierung mit erhaltenem Seitenverhältnis und schwarzem Padding auf 512 × 512; RGB-Wiederholung und ImageNet-Normalisierung vor dem Encoder.
- Insgesamt 927.585 Parameter. Beim partiellen Nachtraining 639.073 trainierbare Parameter; BatchNorm bleibt eingefroren.
- Drei Nachtrainingsepochen ausgewertet, einschließlich unverändertem Ausgangsstand Epoche 0. Gewählt wurde Epoche 1 ausschließlich nach NLL auf den Modellwahldaten. Ein Trainingsseed; keine belegte optimale Architektur oder optimale Trainingsdauer.
- [Trainingscode](../experiments/original_512.py), [Studienprotokoll](../configs/original_512.json), [gespeicherte Ergebnisse und Modellwahl](../reports/original_512/summary.json).

## Gemessene Leistung und Grenzen

| Modell / Auswertung | AUROC | Accuracy | Sensitivität | Spezifität |
|---|---:|---:|---:|---:|
| THORAXIS 512-R1, interner Test mit 618 Bildern | 0,9669 | 82,7 % | 99,5 % | 54,5 % |
| THORAXIS 224-Linear, extern gegen No finding | 0,6729 | 50,9 % | 78,9 % | 47,5 % |
| THORAXIS 224-Linear, extern gegen andere Labels | 0,5480 | 39,1 % | 78,9 % | 29,5 % |

Die externen Zeilen betreffen ein anderes THORAXIS-Modell und eine fallkontrollartige Stichprobe mit textbasierten NIH-/ChestMNIST-Labels. Sie sind keine externe Prüfung von THORAXIS 512-R1. No finding ist keine unabhängig bestätigte Gesundheit. Fehlende Patientenverknüpfungen und mögliche NIH-/RSNA-Überlappung begrenzen die Unabhängigkeit. Die 512-R1-Sensitivität geht mit vielen falsch positiven Normalbildern einher. Eine Vorverarbeitungsänderung kann die Entscheidungen stark verändern; Modellkonfidenz ist keine nachgewiesen kalibrierte Erkrankungswahrscheinlichkeit.

[Interne Studie](../reports/original_512/README.md) · [Vorverarbeitungsprüfung](../reports/original_512/preprocessing/README.md) · [Externe Studie](../reports/external_chest224/README.md).

## Identität und technischer Export

Der lokale Forschungscheckpoint von THORAXIS 512-R1 ist `data/original_xray/mobile512_research.pt`, SHA-256:

`ba497699914926742ee07592297f803925e835c0e2491e2c284eb008804f01ce`

Die beiden Linear-Varianten sind durch den ursprünglichen Torchvision-Encoder und die jeweiligen Kopfgewichte in [probe_heads.npz](../reports/original_512/probe_heads.npz) definiert. Der historische Dateiname des Checkpoints und die Ergebnis-IDs bleiben bestehen.

Ein Core-ML-Export und synthetische Referenzen sind vorbereitet. Apple-Laufzeit, Xcode-Build und Bildvorverarbeitung auf Apple sind weiterhin ungeprüft; siehe [SwiftUI-Anleitung](../ios/ResearchPrototype/README.md). THORAXIS ist ein Forschungsmodell zur Untersuchung von Zuverlässigkeit, keine klinisch freigegebene Diagnoseanwendung. Die [Lizenzabgrenzung](LICENSING.md) für Code, Daten und fremde Gewichte gilt weiter.

Training, Auswertung, Dokumentation und Namenseinführung wurden mit KI-Assistenz durchgeführt. Die Benennung behauptet keine unabhängig belegte persönliche Urheberschaft an den zugrunde liegenden Architekturen oder am Vortraining.
