# Externe Röntgendaten: Machbarkeit und Grenzen

Geprüft am 8. November 2026 anhand der verlinkten öffentlichen Originaldokumentation. Keine geschützten Bilddaten heruntergeladen, keine Nutzungsvereinbarung angenommen und kein Zugang beantragt. Die Prüfung belegt Eigenschaften der Dokumentation, keine funktionsfähige Downloadberechtigung.

| Kandidat | Was die Dokumentation bietet | Hürde für unser Projekt |
|---|---|---|
| [MIMIC-CXR-JPG 2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) | Röntgenbilder, strukturierte aus Befundtexten gewonnene Labels, `subject_id`, `study_id` und Splitdateien. Labelwerte und fehlende/unsichere Angaben sind zu beachten. | Zugang laut PhysioNet nur nach Credentialing, vorgeschriebener Schulung und Data Use Agreement. Labeldefinition und Population unterscheiden sich von PneumoniaMNIST. Patientenschlüssel machen eine gruppierte Auswertung grundsätzlich prüfbar. |
| [VinDr-CXR 1.0.0](https://physionet.org/content/vindr-cxr/1.0.0/) | 18.000 Bilder aus zwei vietnamesischen Krankenhäusern, 15.000 Training / 3.000 Test; Radiologenannotation. Globales Pneumonie-Label vorhanden. | Erwachsene; pädiatrische und minderwertige Bilder wurden bei der Kuratierung ausgeschlossen. Zugang erfordert ebenfalls Credentialing, Schulung und DUA. Originale Patienten-IDs wurden entfernt; eine stabile gruppierbare Patientenkennung muss vor Auswahl eigens geprüft werden. |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | Laut Projektseite 224.316 Röntgenbilder von 65.240 Patienten; 14 Beobachtungen mit positiven, negativen und unsicheren Labels, automatisch aus Berichten gewonnen. Verweist zum Download auf Stanford AIMI. | Die hier geprüfte Übersichtsseite allein bestätigt weder aktuelle Downloadbedingungen noch unsere Zugangsberechtigung. Eine separate Prüfung der tatsächlichen Bereitstellung ist offen. Die Benchmarkaufgabe ist nicht mit pädiatrischer Normal/Pneumonie-Klassifikation identisch. |

## Entscheidung

**Keinen dieser Datensätze automatisch als gleichwertigen externen Test behandeln.** Ein Modell für Kinderbilder auf einer anderen Population mit anderen Labelregeln zu testen kann ein interessantes Transferexperiment sein, muss aber genau so benannt werden. Es darf weder Populationswechsel noch Labelwechsel mit Bildqualitätsverschlechterung vermischen.

MIMIC-CXR-JPG ist ein Kandidat für eine künftig patientengruppierte Studie, weil IDs und strukturierte Labels dokumentiert sind. Das ist keine endgültige Empfehlung: Zugang, Datenumfang, Nutzungserlaubnis, Projektionen, Alter und Labelregeln müssen vor Umsetzung geklärt sein. VinDr kann radiologische Annotationen beisteuern, hat aber eine andere kuratierte Population. Für die aktuelle kostenlose, zugangsunabhängige Arbeit bleibt MedMNIST ein Entwicklungsbenchmark.

## Vor einer externen Auswertung verbindlich festlegen

1. Zielpopulation, Projektion (frontal/lateral) und genaue Zielvariable. „Pneumonia“ ist kein zwischen Quellen automatisch identisches klinisches Goldstandardlabel.
2. Umgang mit unsicheren/fehlenden Labels; keine stillschweigende Umwandlung in „normal“. Positive Pneumonie gegen „keine Pneumonie“ ist eine andere Aufgabe als gegen ausschließlich unauffällige Bilder.
3. Nachweisbare Gruppierung nach Patienten und feste Auswahl der Fälle. Bilder derselben Untersuchung nicht als unabhängige Personen zählen.
4. Ob das alte Modell eingefroren extern getestet oder ein neues Modell entwickelt wird. Bei Entwicklung auf der neuen Quelle braucht es erneut getrennte Daten.
5. Speicher-, Nutzungs- und Veröffentlichungsregeln. Rohdaten und geschützte Beispiele dürfen nicht pauschal nach GitHub kopiert werden.

Zugangsanforderungen sind keine Aufforderung, geheime Zugangsdaten im Chat mitzuteilen. Erst nach einer fachlichen Datensatzauswahl ist eine konkrete Zugangsentscheidung sinnvoll. Bis dahin bleiben die bestehenden lokalen Analysen unabhängig ausführbar.

Quelle und Abrufnachweise: [external_sources.json](../../reports/selective_review/external_sources.json).
