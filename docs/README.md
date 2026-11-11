# Dokumentation

Aktueller Stand: [**Forschungsbericht, 6 Seiten (PDF)**](../reports/followup/research_report.pdf) · [Textfassung](../reports/followup/research_report.md).

- [224px-Mischaugmentation: gezielte Blur-Intervention mit zusätzlicher Kontrolle](../reports/high_resolution/mixed_augmentation/README.md)
- [224px-Störstärken: Rauschen, Unschärfe und gemeinsame Unsicherheit](../reports/high_resolution/severity/README.md)
- [224px-Intervention: Rauschaugmentation des linearen Kopfes](../reports/high_resolution/augmentation/README.md)
- [224px-Robustheit und Filterwirkung der Verkleinerung](../reports/high_resolution/robustness/README.md)
- [224px versus 28px-Information: kontrollierter Auflösungsvergleich](../reports/high_resolution/README.md)
- [Temperature Scaling: negative Übertragung auf den Test](../reports/temperature_scaling/README.md)
- [Augmentationsvergleich mit fünf Seeds](../reports/five_seed/README.md)
- [Externer Test und AP/PA-Standardisierung](../reports/external/README.md)

Die [historische Paper Summary](PAPER_SUMMARY.md) beschreibt den Stand vor diesen Erweiterungen.

- [Code- und Datenlizenz](LICENSING.md)
- [Prüfprotokolle zum Versionsstand v1.0.0](releases/v1.0.0/README.md)
- [Archiv: Berichte und Versuchsverzeichnis](archive/EXPERIMENTS.md)
- [Archiv: Daten-Audit](archive/AUDIT.md) und [Literatur](archive/LITERATURE.md)

Die historischen Berichte und Protokolle liegen zusammen in `archive/`. Die gespeicherten Ergebnisse und Modelle behalten ihre ursprünglichen Pfade. `research/` enthält nur fünf kurze Verweise für die Links im unveränderten Modellverzeichnis; die Berichte selbst liegen im Archiv. Historische Auswerteprogramme können diesen Pfad weiterhin in isolierten Prüfkopien verwenden.

Die separat reproduzierte Grafik lässt sich mit `python scripts/verify_overview.py --output /tmp/medical-ai-overview` prüfen. Der Prüfer schreibt nur in den gewählten neuen Ausgabeordner.

## Originalbilder und App-Export

- [512px, partielles Encodertraining und ResNet-50: Methoden und Ergebnisse](../reports/original_512/README.md)
- [Neue Forschungsnotiz als PDF](../reports/original_512/research_note.pdf)
- [SwiftUI-Tensorprüfer und Core-ML-Export](../ios/ResearchPrototype/README.md)

- [Herkunftsaudit der Normalbilder: Bildgrößen, Erscheinungsmerkmale und gruppierte Kreuzvalidierung](../reports/original_512/source_audit/README.md)

- [512px-Vorverarbeitung: gepaarte Stressanalyse und synthetische App-Referenzen](../reports/original_512/preprocessing/README.md)
