# Medical AI Reliability Lab

Wie verändern Bildstörungen die Genauigkeit und die Konfidenz eines medizinischen Bildklassifikators?

Status: Dateneinstieg implementiert; noch kein trainiertes Modell und keine wissenschaftlichen Ergebnisse zur Zuverlässigkeit.

## Start ohne bezahlte Infrastruktur

[Notebook in Google Colab öffnen](https://colab.research.google.com/github/paploooov/medical-ai-reliability-lab/blob/main/notebooks/01_data_exploration.ipynb). Kostenlose CPU-Laufzeit verwenden. Google Cloud Console, Kreditkarte und Probeabo sind nicht erforderlich. Colab-Verfügbarkeit ist nicht garantiert; lokal funktioniert derselbe Einstieg.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/inspect_data.py
```

Das Skript lädt etwa 4 MB, prüft die offizielle Prüfsumme, validiert alle Splitgrößen und erstellt `reports/training_examples.png` sowie `reports/dataset_summary.json`. Die Bildauswahl ist mit Seed 42 reproduzierbar. Rohdaten bleiben außerhalb von Git.

## Orientierung

- `notebooks/`: deutsch erklärter Colab-Einstieg
- `scripts/`: lokal ausführbarer Code
- `docs/PROTOCOL.md`: Forschungsfrage und geplante Auswertung
- `docs/STATUS.md`: Übergabe und nächste Schritte
- `reports/`: überprüfbare Artefakte

## Quellen und Grenzen

[MedMNIST](https://medmnist.com/), [offizieller Code und Metadaten](https://github.com/MedMNIST/MedMNIST), [Datenarchiv](https://zenodo.org/records/10519652). PneumoniaMNIST: CC BY 4.0 laut offiziellen Metadaten; Attribution bei Weiterverwendung erhalten. Bitte MedMNIST v2 (Yang et al., Scientific Data 2023, DOI: 10.1038/s41597-022-01721-8) und die dort genannte Originaldatenquelle zitieren.

Pädiatrische Röntgenbilder in sehr geringer Auflösung; keine klinische Anwendung. Synthetische Störungen belegen keinen Krankenhaus- oder Gerätewechsel. Die offiziellen Splits allein erlauben ohne Patientenmetadaten keine unabhängige Prüfung der Patiententrennung.
