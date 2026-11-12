# THORAXIS: 30 Trainingsläufe ohne Testinferenz

Protokoll: `c44117a`. Status: **running**, **30/30** Modelle abgeschlossen. Stand (UTC): 2026-11-12T20:15:29.919215+00:00.

Fünf offizielle Referenzmodelle und fünf gemischte Aufteilungen mit jeweils denselben Seeds 17, 42, 83, 123 und 2026. Alle Modelle starten neu vom ursprünglichen ImageNet-Encoder. Der frühere technische Probelauf bleibt getrennt. Training, Modellwahl und Kalibrierung verwenden nur die eigenen Entwicklungsdaten. Der medizinische Kopf erhält keine Testzeilen.

Die [vollständige Tabelle aller 30 Läufe](../reports/original_512/thoraxis_training_30.csv) enthält gewählte Epoche, Modellwahl-NLL, Laufzeit, Checkpoint-SHA-256 und lokalen Gewichtspfad. [Logs aller Versuche](../reports/original_512/thoraxis_training_30_logs.json) enthalten Epochenverlauf und Abbruchgründe. Bisher **0** fehlgeschlagene/unterbrochene Versuche; Summe der erfolgreichen Laufzeiten: **4.08 Stunden**. Abgebrochene Versuche werden erhalten und mit demselben Seed wiederholt, höchstens dreimal insgesamt; danach wird der offene Lauf markiert.

Gewichte liegen nach bestehender Regel im ignorierten Verzeichnis `data/thoraxis_source_shift/training_batch/`; Logs und Hashes werden nach jedem Lauf committet. Keine Gewichte werden hochgeladen. Der Auftrag läuft seriell im Hintergrund, speichert seinen Fortschritt und beendet sich nach dem Paket. Wiederaufnahme: mit derselben gesperrten Python-Umgebung `python scripts/run_thoraxis_batch.py`; vorhandene erfolgreiche Checkpoints werden anhand ihres Hashes geprüft und nicht neu trainiert.

## Grenzen

**Keine Testinferenz und keine neue Aussage über Testleistung oder Kausalität.** Die Modellwahl-NLL beschreibt ausschließlich die Selection-Menge. Lokale Gewichte sind noch kein öffentlich abrufbares Release. Bei Eingabe-/Hashfehlern stoppt der Auftrag; ein Prozessabbruch erfordert Wiederaufnahme mit dem obigen Befehl. Details eines möglichen Blockers stehen in den Logs.
