# Roadmap

Explorative Studie, Stand 10. November 2026. Die folgenden Schritte sind teils durchgeführt; offene Anteile sind ausdrücklich benannt. Gewichte, Daten und Ergebnisse des bisherigen Stands bleiben erhalten. Reihenfolge:

1. **Kalibrierung (teilweise):** ECE und Reliability Diagrams ohne Temperature Scaling liegen im [externen Bericht](reports/external/README.md) vor. Offen: ECE und Reliability Diagrams für Originalbilder und alle Störungen, vor und nach Temperature Scaling. Bin-Regel und Auswertung vorher festlegen; Temperatur ausschließlich auf Kalibrierungsdaten bestimmen, nicht auf dem Testset.
2. **Selektive Vorhersage (teilweise):** Externe Risk-Coverage-Kurven und das vorab fixierte [Protokoll](reports/external/PROTOCOL.md) liegen vor. Weiterhin geplant: Risk-Coverage-Kurven als Primärendpunkt über alle Störungen. Ablehnungsregel, Auswertung und Zusammenfassung vor der nächsten Testauswertung im Repository fixieren. Der bestehende PneumoniaMNIST-Test wurde bereits angesehen und wird dadurch nicht nachträglich zu einem unberührten Testset.
3. **Externer Test (durchgeführt, Rechteprüfung offen):** Eingefrorene 28px-/64px-Modelle auf der fixierten RSNA-Stichprobe ausgewertet; [Ergebnisse einschließlich AP/PA-Nachanalyse](reports/external/README.md). Keine Anpassung anhand externer Testergebnisse. Original-Lizenzbedingungen und Weitergaberechte des HF-Spiegels bleiben zu klären. Patientenunabhängigkeit über verschiedene RSNA-IDs ist nicht nachgewiesen.
4. **Kontrastvergleich:** Mindestens fünf Seeds und eine zweite Architektur, ResNet-18 auf MedMNIST+ 64px. Als ergänzenden Kontrast ausweisen; Architektur und Auflösung ändern sich gemeinsam.
5. **Bericht:** Vier bis sechs Seiten im Preprint-Stil: Fragestellung, Methode, Ergebnisse und Grenzen. Eine Hauptabbildung verbindet Risk-Coverage und Nullvektor-Effekt. Nur tatsächlich vorliegende Ergebnisse aufnehmen.
6. **Eigenleistung & Werkzeuge:** In der README offenlegen, welche Teile mit KI-Assistenz entstanden sind und welche Entscheidungen und Prüfungen vom Projektverantwortlichen selbst vorgenommen wurden. Persönliche Beiträge gemeinsam klären, nicht aus dem Repository-Verlauf vermuten.

Weniger, dafür inhaltlich zusammenhängende Commits mit klaren Messages. Keine zusätzlichen Dokumentationsseiten ohne neues Ergebnis; diese Roadmap ist der Plan, kein Ergebnisbericht.
