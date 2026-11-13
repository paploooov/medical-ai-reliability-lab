# THORAXIS: H1 und H2 — einmalige Testauswertung

12. November 2026 · Protokoll `c44117a` · 30 fertige Modelle · keine Änderung an Gewichten, Schwellen oder Auswahlregeln.

## Beobachtung

Bei genau 80 % Abdeckung beträgt der mittlere Fehler **11,29 % offiziell** und **0,63 % gemischt**. Der gepaarte Unterschied ist **+10,66 Prozentpunkte**. Alle 25 Kontraste sind positiv ([5×5-Tabelle](../reports/original_512/thoraxis_30/h1_contrasts.csv)). Die getrennten 95%-Bereiche sind **Seed 10,07–11,26**, **Split 10,13–11,20** und **Gruppen-Bootstrap 7,29–14,41 Prozentpunkte**. Damit ist die vorher festgelegte **beschreibende Erfolgsregel erfüllt**. Die fünf offiziellen Referenzen wurden wiederverwendet, nicht als 25 unabhängige Modelle gezählt. Alle 2.000 Bootstrap-Ziehungen waren für die Hauptzahl definiert.

Die Spezifität beträgt im offiziellen Arm **93,65 % auf Kalibrierung und 49,00 % auf Test**, im gemischten Arm **92,48 % und 92,33 %**. Das sind die neuen Modellmittel, nicht die historischen Werte des einzelnen R1-Modells.

Die feste Konfidenzablehnung fängt **76,1 % / 89,8 %** der Fehlalarme ab (offiziell/gemischt). Sie verwirft aber auch **68,0 % / 27,1 %** der richtigen Normalbefunde. Ihre tatsächliche Gesamtannahme beträgt **68,5 % / 78,6 %**; die Rangkennzahl bei 80 % ersetzt diese feste Schwelle nicht.

| Klasse | Annahme offiziell / gemischt | Fehler unter Angenommenen offiziell / gemischt |
|---|---:|---:|
| Normal | 27,9 % / 68,1 % | **43,77 % / 1,12 %** |
| Pneumonie | 92,8 % / 84,8 % | 0,11 % / 0,28 % |

Im gemischten Test bleiben ursprünglich offizielle Testfälle schwieriger: Spezifität **72,4 %** gegenüber **95,7 %** bei ursprünglichen Trainingsfällen. Je Split sind dies **66–94** beziehungsweise **524–552** Bilder. Alle Angaben sind gleichgewichtete Modellmittel; Einzelzahlen, Nenner und Herkunft/Klasse stehen in den [Detaildaten](../reports/original_512/thoraxis_30/h2_metrics.json).

## Mögliche Erklärung

Die Ergebnisse passen zu einer Übertragungslücke zwischen den ursprünglichen Beständen. Mischen bringt beide Bestände ins Training. Konfidenz hilft bei der Ablehnung, schützt aber gerade Normalbilder im offiziellen Arm unzureichend.

## Fehlender Nachweis und Grenzen

Kein isolierter Herkunftseffekt, kein kausaler Shortcut-Nachweis und keine klinische Validierung: Trainings- und Testzusammensetzung ändern sich gemeinsam. Testbilder waren bereits aus früheren Studien bekannt; Gruppen sind Dateinamen-Gruppen, keine bestätigten Patienten. Die drei Unsicherheitsbereiche bilden kein gemeinsames Gesamtintervall. Null beobachtete Fehler bedeuten kein Nullrisiko; ergänzende Gruppengrenzen stehen in den Detaildaten.

Gewählte Epochen **0/1/2/3: 1/4/9/16 Modelle**. Bei **16/30 (53,3 %)** lag die Wahl an der Budgetgrenze; ausreichende Trainingsdauer ist damit nicht nachgewiesen. Es wurde nicht nachtrainiert.

**Chronologie (Berlin, 12.11.):** Protokoll lokal 11:59:41, Push 11:59:44. Splits lokal 12:14:36, Laufplan lokal 17:01:44; beide erst **18:29:13** auf GitHub gepusht, nach Trainingsbeginn 17:01:58. Alle 30 Checkpoints: Push **21:23:41** (`8592f3a`), Inferenzstart **21:26:26**. Push-Zeiten stammen aus dem lokalen Reflog, nicht aus unabhängigen GitHub-Serverzeitstempeln; der Remote-Hash wurde live bestätigt. Keine externe Präregistrierung. [Zeitbelege](../reports/original_512/thoraxis_30/pretest_publication.json). Die nachträglich abgerufene [GitHub-Activity-API](../reports/original_512/thoraxis_30/github_push_times.json) bestätigt die Push-Zeiten serverseitig, wobei `5305e2c` und `dad8107` im gemeinsamen Push auf `7041378` enthalten waren.

[Grafik 1: H1](../reports/original_512/thoraxis_30/h1_contrasts.svg) · [Grafik 2: H2](../reports/original_512/thoraxis_30/h2_rejection.svg) · [Getrennte Streuungen](../reports/original_512/thoraxis_30/h1_results.json) · [H2-Bootstrap](../reports/original_512/thoraxis_30/h2_bootstrap.json) · [Konfidenzverteilungen](../reports/original_512/thoraxis_30/h2_confidence.json).

Unabhängige Nachrechnung: `python scripts/check_thoraxis_primary.py` bestätigt die Hauptzahl und alle 25 Kontraste exakt ([Prüfergebnis](../reports/original_512/thoraxis_30/independent_check.json)). Vorhersagen samt Hashes: [Manifest](../reports/original_512/thoraxis_30/prediction_manifest.json).
