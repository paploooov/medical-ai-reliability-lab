# THORAXIS

**Forschungsmodelle für Pneumonie-Klassifikation und Zuverlässigkeit auf Thorax-Röntgenbildern.** THORAXIS 512-R1 basiert auf MobileNetV3-Small mit ImageNet-Vortraining; im Projekt wurden die Klassifikationsschicht und Teile des Encoders trainiert. Architektur und Vortraining stammen aus Torchvision.

| Gespeicherter Forschungsstand | Wichtigstes Ergebnis |
|---|---|
| THORAXIS 512-R1, interner Test | AUROC **0,967**, Accuracy **82,7 %**, Spezifität **54,5 %** |
| THORAXIS 224-Linear, externer Test | AUROC **0,673** gegen textbasiertes „No finding“ |
| 512px-Vorverarbeitung | Center Crop verändert **17,5 %** der Entscheidungen |

Die externen Werte validieren ein anderes Modell als 512-R1. Patientenunabhängigkeit und klinische Eignung sind nicht nachgewiesen; hohe Konfidenz schützt extern nicht zuverlässig vor Fehlern.

**[Alle bisherigen Teilstudien: VORARBEITEN](docs/VORARBEITEN.md)** · [Modellkarte](docs/THORAXIS_MODEL_CARD.md) · [Roadmap](ROADMAP.md)

```bash
make reproduce-512
```

Linux/Python 3.12, CPU: Training, Modellwahl, Kalibrierung und Export; Testauswertung optional und bislang nicht im Reproduktionslauf ausgeführt. Gemessen: **15,17 Minuten, 1,16 GiB Peak-RAM**, mit lokal bereitgestelltem Originalarchiv; [Anleitung und Grenzen](docs/reproduction-512/README.md).

**Alte Berichte:** [Ergebnisberichte und Daten](reports/) · [Historisches Archiv](docs/archive/) · [Historische Verweisdateien](docs/research/).

[Zitieren](CITATION.cff) · [Lizenzen und Gewichte-Freigabe](docs/LICENSING.md) · [Checkpoint-Veröffentlichungsprüfung](docs/reproduction-512/RELEASE_REVIEW.md). Entwicklung und Dokumentation mit KI-Assistenz; kein klinisch freigegebenes Diagnoseprodukt.
