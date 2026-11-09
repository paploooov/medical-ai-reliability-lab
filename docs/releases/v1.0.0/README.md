# Abschlussstand v1.0.0 – PneumoniaMNIST, Versuche 01–10

Die Versuchsreihe ist abgeschlossen. Dieser Stand ergänzt die Gesamtsynthese und einen englischen Abstract, enthält aber keine neuen Experimente, Gewichte oder wissenschaftlichen Ergebnisberechnungen. Die erneute Ausführung bestehender Verifikationen dient ausschließlich der Prüfung des festgehaltenen Forschungsstands. Nicht unabhängig bestätigte Aussagen und verbleibende Grenzen bleiben ausgewiesen; „abgeschlossen“ bedeutet nicht begutachtet oder klinisch validiert.

## Inhaltliche Präzisierung

Der nachgewiesene Zusammenhang verbindet Null-Merkmalsvektor, Bias-Logit, Konfidenz-Gleichstände und das Versagen der untersuchten Quantil-Annahmeregel. Die Umkehrung gilt nicht: Ein nichtleerer Merkmalsvektor garantiert keine verlässliche Vorhersage. Nullvektor-Ausschluss kann korrekte konstante Vorhersagen entfernen und dadurch die verbleibende Fehlerrate erhöhen.

„Mehrheitsklassen-Bias“ wäre für die fünf betroffenen Modelle unzutreffend: Ihre negativen Bias-Werte erzeugen „normal“, während im offiziellen Datensatz Pneumonie überwiegt. Die Synthese unterscheidet deshalb einen gelernten konstanten Bias-Output von einem empirischen Mehrheitsklassen-Prädiktor. Es wurden keine Modelle verändert, um diese redaktionelle Aussage zu stützen.

## Abschlussprüfung

- Alle **43 Unit-Tests** bestanden: [vollständiges Testprotokoll](project_checks/00.log).
- Audit, Verifikationen zu Auflösung, selektiver Vorhersage und Kalibrierung sowie Budget-, Augmentations- und Auflösungsprüfungen bestanden: [Befehle und Exit-Codes](project_checks/verification.json).
- Versuch 09 erneut separat verifiziert: 72 Entscheidungszeilen, deterministische Wiederholung einschließlich Bootstrap, [Prüfdatei](mechanistic_ties_verification.json) und [Log](mechanistic_ties.log).
- Versuch 10 erneut separat verifiziert: 48 Ergebniszeilen, 129 Arrays und vier Abbildungsdateien, [Prüfdatei](transfer_verification.json) und [Log](transfer.log).
- **193 vorhandene Ergebnis- und Modellartefakte** unverändert. Das schließt sämtliche zuvor gesicherten 166 und die hinzugekommenen Versuch-10-Artefakte ein. [Vollständiges SHA-256-Manifest](artifacts.sha256.json) und [Abschluss-Prüfartefakt](verification.json).

Alle Programme, die vorhandene Prüfdateien schreiben, wurden in isolierten Kopien beziehungsweise auf temporären Kopien der betreffenden Berichtsordner ausgeführt. Kein früheres `verification.json` wurde ersetzt. Das Original-Arbeitsverzeichnis erhielt nur redaktionelle Änderungen und diese zusätzlichen Abschlussprotokolle unter `docs/releases/`.

## Prüfung erneut ausführen

Aus dem Repository-Root mit der dokumentierten Python-Umgebung und den vorhandenen Datenarchiven:

```bash
RELEASE_CHECK_DIR=$(mktemp -d)
python experiments/verify_project.py --output "$RELEASE_CHECK_DIR/project_checks"
cp -a reports/mechanistic_ties "$RELEASE_CHECK_DIR/mechanistic_ties"
rm "$RELEASE_CHECK_DIR/mechanistic_ties/verification.json"
python experiments/verify_mechanistic_ties.py --output "$RELEASE_CHECK_DIR/mechanistic_ties"
cp -a reports/transfer "$RELEASE_CHECK_DIR/transfer"
rm "$RELEASE_CHECK_DIR/transfer/verification.json"
python experiments/verify_transfer.py --output "$RELEASE_CHECK_DIR/transfer"
```

Die entfernten Prüfdateien liegen ausschließlich in den gerade angelegten temporären Kopien. `verify_project.py` führt bereits die vollständige Unit-Testsuite aus. Die ursprünglichen Artefakte lassen sich zusätzlich direkt prüfen:

```python
from pathlib import Path
import hashlib
import json

manifest = json.loads(Path("docs/releases/v1.0.0/artifacts.sha256.json").read_text())
for name, expected in manifest.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print(f"{len(manifest)} Artefakte unverändert")
```

## Versionierung

Der annotierte Git-Tag `v1.0.0` bezeichnet den redaktionellen Abschlusscommit einschließlich dieser Prüfprotokolle. Eine mögliche spätere externe Folgestudie gehört nicht mehr zur abgeschlossenen Versuchsreihe 01–10. Es wird keine neue medizinische Freigabe oder Sicherheitsgarantie mit dieser Versionsnummer verbunden.
