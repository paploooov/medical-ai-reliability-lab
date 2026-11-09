# Versuch 09 – Ursache der Gleichstände und Folgen alternativer Annahmeregeln

Nachanalyse vom 9. November 2026. Sechs eingefrorene 28×28-CNNs aus Versuch 06/07; dieselben 262 Kalibrierungsfälle und 621 Testrepräsentanten wie in Versuch 08. Kein neues Training, keine Auswahl eines günstigeren Hash-Salts, keine unabhängige Validierung. Die früheren Gewichte und Ergebnisdateien bleiben unverändert.

## Mechanismus

Der letzte Klassifikator berechnet `z = wᵀh + b`, danach `p = sigmoid(z)` und `confidence = max(p, 1-p)`. Wenn der 32-dimensionale letzte ReLU-Merkmalsvektor vollständig null ist, gilt unabhängig vom Bild exakt `z = b`. Diese vollständige Nullaktivierung erklärt sämtliche Gleichstände an den fünf betroffenen Kalibrierungsschwellen:

| Modell | Seed | Konfidenzschwelle | Fälle genau auf der Schwelle | Davon Nullvektoren |
|---|---:|---:|---:|---:|
| Kontrolle | 17 | 0,940302432 | 0 | 0 |
| Kontrolle | 42 | 0,768156052 | 51 | 51 |
| Kontrolle | 83 | 0,746399522 | 51 | 51 |
| Augmentation | 17 | 0,737012982 | 51 | 51 |
| Augmentation | 42 | 0,767967582 | 50 | 50 |
| Augmentation | 83 | 0,753490090 | 47 | 47 |

Insgesamt 250 Modell-Fall-Paare, nicht 250 verschiedene Patienten. Die erneute Inferenz reproduziert alle gespeicherten Kalibrierungs- und primären Testwahrscheinlichkeiten exakt. Für jeden Nullvektor wird die exakte Gleichheit von Logit und gespeichertem Bias geprüft. Die Aussage betrifft diese Endschicht; nicht jedes frühere Merkmal muss ebenfalls null sein. Sie erklärt die Punktmasse, aber nicht vollständig, warum das Training gerade diese ReLU-Region erzeugt hat.

**Keine Sättigung an der Schwelle:** Kein Kalibrierungsfall hat Konfidenz 1,0. Bei den augmentierten Modellen treten auf Original-Testbildern insgesamt zwei, unter Rauschen vier Modell-Fall-Paare mit Konfidenz 1,0 auf; diese vereinzelte Float-Sättigung ist ein anderer Effekt. **Keine diskrete Wahrscheinlichkeitskalibrierung:** Versuch 08 verwendet rohe Sigmoid-Ausgaben und das linear interpolierte 20%-Quantil als Annahmeschwelle, weder Isotonic Regression noch Binning oder Temperature Scaling. Die Quantilregel erzeugt die Punktmasse nicht, akzeptiert sie mit `>=` aber vollständig.

## Annahmeregeln und Rundung

1. **Original:** Unveränderte Regel aus Versuch 08, `confidence >= quantile(calibration, 0.2)`.
2. **Tie-Breaking:** Kalibrierungsfälle nach absteigender Konfidenz sortieren, exakte Gleichstände nach SHA-256 von `study09-v1:split:original_index` aufsteigend. Die Grenze liegt bei `floor(4*n/5)` Fällen. Bei 262 Fällen sind das **209/262 = 79,770992 %**. Exakt 80 % wären 209,6 Fälle und sind mit binären Einzelentscheidungen unmöglich. Auch 80 % von 621 sind keine ganze Fallzahl. Fraktionale Fallgewichte wären eine andere Auswertung, keine binäre Annahmeregel.
3. **Nullvektoren zurückhalten:** Originalschwelle beibehalten und zusätzlich `h != 0` verlangen. Keine Auffüllung und keine erneute Schwellenanpassung.
4. **Kombination:** Tie-Breaking plus Nullvektor-Ausschluss, ebenfalls ohne Auffüllung.

Die aus der Kalibrierung gewonnene Konfidenz-/Hash-Grenze wird auf Testfälle übertragen. Testlabels gehen nicht in die Entscheidung ein; die Testquote wird nicht auf 80 % gezwungen. Die Wahl eines einzelnen Hash-Salts ist dennoch eine zusätzliche Festlegung: Andere Ordnungen innerhalb der ununterscheidbaren Gruppe können andere Fehleranteile ergeben. Hier wurde keine Sensitivitätsanalyse über Salts vorgenommen. Die primäre Teststörung bleibt Gaußrauschen mit SD 0,1, Seed 42 und Begrenzung auf [0,1].

## Resultate unter Rauschen

Zellen zeigen **akzeptierte Fälle / davon Fehler**, jeweils von 621 Testfällen. Alle Original-, Kalibrierungs- und Rauschbedingungen sowie vier Regeln stehen in der [vollständigen Tabelle](../../reports/mechanistic_ties/outcomes.csv).

| Modell | Seed | Original | Tie-Breaking | Nullvektoren zurückhalten |
|---|---:|---:|---:|---:|
| Kontrolle | 17 | 425 / 203 | 424 / 202 | 425 / 203 |
| Kontrolle | 42 | 602 / 369 | 279 / 173 | 2 / 2 |
| Kontrolle | 83 | 614 / 381 | 178 / 106 | 0 / undefiniert |
| Augmentation | 17 | 505 / 60 | 456 / 59 | 420 / 58 |
| Augmentation | 42 | 503 / 58 | 443 / 57 | 414 / 55 |
| Augmentation | 83 | 523 / 75 | 472 / 75 | 451 / 75 |

Tie-Breaking behebt die Überakzeptanz auf dem Kalibrierungssatz, nicht den Informationsverlust. Die Kontrollmodelle mit Seed 42 und 83 akzeptieren weniger Rauschfälle, bleiben darunter aber häufig falsch. Nullvektor-Ausschluss entfernt auch korrekte Vorhersagen: Bei den augmentierten Modellen steigt die Fehlerrate unter den verbleibenden akzeptierten Rauschfällen von 11,9/11,5/14,3 % auf 13,8/13,3/16,6 %. Der Ausschluss ist somit ein diagnostischer Eingriff und kein nachgewiesener Sicherheitsgewinn.

## Warum bleiben Teilmengen leer?

Die 54 leeren Ziehungen aus Versuch 08 betreffen Kontrolle/Seed 83 unter Rauschen. In **allen 54** liegt die neu gezogene Kalibrierungsschwelle strikt über der höchsten Konfidenz des gesamten Rauschtestsatzes. Das Problem entsteht also schon vor der Testwiederziehung; dort fehlen nicht lediglich zufällig einige hochkonfidente Fälle. Wenn die Kalibrierungsziehung die Quantilgrenze oberhalb der Bias-Punktmasse verschiebt, verschwindet die zuvor akzeptierte Rauschgruppe vollständig.

Alle 2.000 gespeicherten Kalibrierungs-/Testziehungen wurden für jede Regel wiederverwendet. Bootstrap-Kopien eines Kalibrierungsfalls erhalten einen festen Vorkommenssuffix zur eindeutigen Rangfolge; wiederholte Testfälle behalten dieselbe Entscheidung. Das ist eine dokumentierte Konvention für diese empirische Sensitivitätsprüfung, keine Garantie nominaler Intervallabdeckung.

| Rauschmodell | Original: leere Ziehungen | Tie-Breaking | Nullvektor-Ausschluss | Kombination |
|---|---:|---:|---:|---:|
| Kontrolle, Seed 42 | 0 | 1 | 271 | 272 |
| Kontrolle, Seed 83 | 54 | 54 | 2.000 | 2.000 |

Bei allen übrigen Modell-/Bedingungskombinationen sind es null. Bei leerer Annahmemenge bleibt die Fehlerrate undefiniert, niemals null. Die Gegenproben rechtfertigen keine neue klinische Annahmeregel und keinen rückwirkend verbesserten Versuch-08-Hauptvergleich.

## Reproduktion und Prüfung

```bash
python experiments/mechanistic_ties.py --output /tmp/study09-new
cp reports/mechanistic_ties/previous_artifacts.json /tmp/study09-new/
python experiments/verify_mechanistic_ties.py --output /tmp/study09-new
python -m unittest discover -s tests -v
python experiments/verify_project.py --output /tmp/project-checks-new
```

Das Auswerteprogramm verweigert das Überschreiben existierender Ergebnisse. CPU-Inferenz mit zwei Threads und deterministischen Torch-Algorithmen; Eingangsdateien, Gewichte und Code sind per SHA-256 erfasst. Der Prüfer vergleicht die Annahmeentscheidungen mit einer separaten skalaren Sortierung und wiederholt die vollständige Berechnung in einem temporären Ordner, einschließlich aller Bootstrap-Ziehungen. Damit wird Determinismus in der dokumentierten Softwareumgebung geprüft, keine Bitgleichheit auf beliebiger Hardware zugesichert.

[Mechanismusdaten](../../reports/mechanistic_ties/diagnostics.csv) · [Bootstrap-Leermengen](../../reports/mechanistic_ties/bootstrap_empty.csv) · [Prüfartefakt](../../reports/mechanistic_ties/verification.json) · [Provenienz](../../reports/mechanistic_ties/provenance.json) · [Prüfsummen der früheren Artefakte](../../reports/mechanistic_ties/previous_artifacts.json) · [Gesamte Projektprüfung](../../reports/mechanistic_ties/project_checks/verification.json).
