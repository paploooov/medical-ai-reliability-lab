# Dokumentation

[**Paper Summary**](PAPER_SUMMARY.md) führt durch Setup, mechanistischen Befund und Grenzen der Intervention.

- [Code- und Datenlizenz](LICENSING.md)
- [Lizenzangaben zu den externen RSNA-Testdaten](DATA_LICENSE_EXTERNAL.md)
- [Prüfprotokolle zum Versionsstand v1.0.0](releases/v1.0.0/README.md)
- [Archiv: Berichte und Versuchsverzeichnis](archive/EXPERIMENTS.md)
- [Archiv: Daten-Audit](archive/AUDIT.md) und [Literatur](archive/LITERATURE.md)

Die historischen Berichte und Protokolle liegen zusammen in `archive/`. Die gespeicherten Ergebnisse und Modelle behalten ihre ursprünglichen Pfade. `research/` enthält nur fünf kurze Verweise für die Links im unveränderten Modellverzeichnis; die Berichte selbst liegen im Archiv. Historische Auswerteprogramme können diesen Pfad weiterhin in isolierten Prüfkopien verwenden.

Die separat reproduzierte Grafik lässt sich mit `python scripts/verify_overview.py --output /tmp/medical-ai-overview` prüfen. Der Prüfer schreibt nur in den gewählten neuen Ausgabeordner.
