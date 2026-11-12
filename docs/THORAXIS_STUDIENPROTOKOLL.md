# THORAXIS: Herkunft und Fehlerablehnung

**Protokoll vom 12. November 2026 · Noch kein Lauf.** Verbindliche Details und geprüfte Quellen stehen in [der Konfiguration](../configs/thoraxis_source_shift_protocol.json).

**Frage:** Hängt der Spezifitätseinbruch von etwa 98 % auf 54,5 % mit der offiziellen Datenaufteilung zusammen, und hält die Konfidenzregel die Fehler zurück?

## Versuchsplan

**H1, primär:** Offizielle Aufteilung gegen eine feste gruppierte Zufallsneuaufteilung derselben 5.371 bereinigten Bilder aus offiziellem Train+Test. Jede Dateinamen-Gruppe bleibt zusammen; beide Tests haben 618 Bilder mit 231 Normal- und 387 Pneumoniebildern. Auch Trainings-, Modellwahl- und Kalibrierungsgrößen samt Klassenanzahlen bleiben gleich. Splitseed: **20261112**; Trainingsseeds: **17, 42, 83, 123, 2026**, insgesamt zehn Läufe. Unlösbare Gruppenzuordnungen führen zum Stopp, nicht zum Probieren günstigerer Splits.

Alle Läufe beginnen beim gleichen ImageNet-Encoder und verwenden das bestehende 512-R1-Training mit drei Epochen. Modellwahl erfolgt nur auf Modellwahldaten. Primär vergleichen wir den Fehler bei genau **80 % Abdeckung**; gleiche Konfidenzen an der Grenze zählen anteilig. Wichtigste Nebenkennzahl ist die **Spezifität bei Schwelle 0,5**. Zusätzlich prüfen wir die auf Kalibrierungsdaten gesetzte Ablehnungsschwelle: Sie bleibt im Test fest und muss dort keine 80 % erreichen.

**H2:** Wir zeigen die Konfidenzverteilung falsch positiver Normalbilder neben richtig erkannten Normalbildern. Akzeptanz, Fehler, zurückgehaltene Fehlalarme und unnötig zurückgehaltene richtige Vorhersagen werden nach wahrer Klasse getrennt mit Zähler und Nenner berichtet.

**H3, explorativ:** Lanczos gegen bilinear bei denselben eingefrorenen Modellen und Bildern über alle fünf Seeds je Aufteilung. Der Effekt wurde bereits beobachtet; wir trainieren dafür keine neuen Modelle.

**Shortcut-Prüfungen:** Ein getrenntes CNN lernt nur bei Normalbildern die ursprüngliche Herkunft, mit fünf gruppierten Außenfolds und innerer Modellwahl. Die medizinischen Modelle werden außerdem mit sichtbarer Lunge beziehungsweise sichtbarem Außenbereich geprüft; flächenidentische Kontrollmasken ergänzen den Vergleich. Dieser Teil beginnt erst nach festgelegter Maskenquelle, anatomischer Qualitätsprüfung und gesperrtem Maskenmanifest.

## Statistik und Grenzen

Seed-Streuung und 2.000 Bootstrap-Wiederziehungen von Testgruppen werden getrennt berichtet; überlappende Gruppen erhalten gemeinsame Wiederziehungsgewichte. Null beobachtete Fehler bedeuten kein Nullrisiko; leere Annahmemengen bleiben undefiniert.

Die Testbilder sind bereits bekannt, Dateinamen sind keine bestätigten Patientenkennungen und die neue Aufteilung verändert Trainings- und Testzusammensetzung. Ein Vorteil nach Mischung stützt eine Herkunftshypothese, beweist aber weder eine Klinikursache noch einen Shortcut; Maskierungsartefakte und nur eine Neuaufteilung begrenzen die Aussage zusätzlich.

## Geplanter Beitrag — fünf Sätze

Wir prüfen eine konkrete Übertragungslücke des eigenen nachtrainierten Modells. Der Vergleich hält Gruppentrennung, Klassenzahlen und Training fest. Wir untersuchen, ob Konfidenzablehnung gerade die falsch positiven Normalbilder erkennt. Herkunfts-CNN und Maskierung liefern ergänzende Hinweise auf mögliche Abkürzungen. Der Beitrag ist eine nachvollziehbare Ursachenprüfung, deren Neuheit und klinische Bedeutung erst durch Ergebnisse und weiteren Literaturvergleich bewertet werden können.

## Geprüfte Literatur

- [Geirhos 2020](https://doi.org/10.1038/s42256-020-00257-z): Begriff und Problem des Shortcut Learning.
- [DeGrave 2021](https://doi.org/10.1038/s42256-021-00338-7): Shortcut-Befunde bei COVID-19-Röntgenmodellen.
- [Zech 2018](https://doi.org/10.1371/journal.pmed.1002683): Herkunft und Übertragung von Pneumoniemodellen.
- [Kermany 2018](https://doi.org/10.1016/j.cell.2018.02.010): Ausgangsarbeit und pädiatrische Röntgendaten.
- [TRIPOD+AI 2024](https://doi.org/10.1136/bmj-2023-078378): transparente Berichterstattung, keine Gütegarantie.

DOI-Metadaten und Abstracts geprüft; Prüfwege stehen in der Konfiguration. **Vor jedem neuen Lauf müssen Protokoll und nötige Manifeste committed sein; Änderungen werden vorher begründet. Diese Etappe endet ohne Testdatenzugriff oder Experiment.**
