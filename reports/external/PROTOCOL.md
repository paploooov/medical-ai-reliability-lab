# Externer Test – vorab festgelegtes Protokoll

Festgelegt vor der ersten Auswertung externer Bilder (Branch `external-test`). Keine Modell-, Schwellen- oder Vorverarbeitungsänderung anhand externer Ergebnisse.

**Daten.** RSNA Pneumonia Detection Challenge (2018), Stage-2-Trainingsbilder (26.684 Thorax-Röntgenaufnahmen Erwachsener aus NIH ChestX-ray14, Labels durch Radiologen-Panels von RSNA/STR). Bezug ohne Login über den öffentlichen Hugging-Face-Spiegel `Baldezo313/rsna-pneumonia-dataset`, fixiert auf Revision `a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4`; jede Datei wird gegen die Git-Blob-Prüfsumme des Spiegels geprüft. Kermany/Guangzhou und davon Abgeleitetes sind ausgeschlossen (Quelle von PneumoniaMNIST).

**Labels.** Positiv (1) = `Lung Opacity` (in der Challenge der Stellvertreter für Pneumonie). Primärer Negativ-Arm (0) = `Normal`. `No Lung Opacity / Not Normal` wird nicht in die Primärauswertung gemischt, sondern als sekundäre Negativ-Stichprobe getrennt berichtet (Falsch-positiv-Rate, Akzeptanz).

**Stichprobe.** Seed 20261109, aus nach `patientId` sortierten Listen ohne Zurücklegen: 1.000 `Lung Opacity` + 1.000 `Normal` (primär, Prävalenz 50 %) und 500 `No Lung Opacity / Not Normal` (sekundär). In RSNA gehört zu jeder `patientId` genau ein Bild; damit ein Bild pro Patient-ID. Die Zuordnung zu NIH-Patienten ist nicht öffentlich, Mehrfachaufnahmen derselben Person über verschiedene IDs sind nicht ausschließbar.

**Vorverarbeitung (wie PneumoniaMNIST/MedMNIST beschrieben).** DICOM-Pixel als 8-Bit-Graustufen (MONOCHROME1 würde invertiert, Nicht-8-Bit min-max skaliert), quadratischer Mittelzuschnitt auf die kurze Kante, Verkleinern auf 28×28 (sekundär 64×64) mit PIL-Bicubic, Speicherung als uint8, Modelleingabe /255 wie im Repo. Die genaue Interpolation von MedMNIST ist nicht dokumentiert; PIL-Bicubic ist eine Annahme.

**Modelle (eingefroren).** Primär: 28px-Kontrolle `models/budget/r28_s{17,42,83}.pt` gegen rauschaugmentierte `models/augmentation/r28_s{17,42,83}.pt` (Versuch 07). Sekundär: 64px-Kontrolle `models/budget/r64_s*.pt`. Gewichte per SHA-256 gegen `run.json` geprüft. Annahmeschwellen unverändert aus `run.json` (20-%-Quantil der Konfidenz auf den Kalibrierungsdaten, Ziel 80 % Akzeptanz).

**Bedingungen.** Wie im Repo: `clean`; `noise` (Gauß, SD 0,1, Seed 42, auf [0,1] begrenzt); `relative_blur` (Gauß-Unschärfe, Sigma = Auflösung/28 Pixel); `contrast` (0,5·(x−0,5)+0,5).

**Kennzahlen.** Accuracy (Schwelle 0,5), AUROC, Sensitivität, Spezifität, ECE (Top-Label-Konfidenz, 15 gleich breite Bins auf [0,1]) mit Reliability Diagram, Annahmerate und Fehler unter akzeptierten Fällen bei der vorhandenen Schwelle, Risk-Coverage-Kurve (Abdeckung 0,05–1,0) mit AURC. Jeweils interner Test (621 Repräsentanten aus `configs/resolution_manifest.json`) und extern.

**Unsicherheit.** 2.000 Bootstrap-Wiederziehungen der Bilder (Seed 2026), gepaart über Arme und Seeds, bedingt auf feste Modelle und Schwellen; Perzentil-95%-Intervalle für Mittelwerte über drei Seeds und für die Differenz Augmentation minus Kontrolle.

**Primärer externer Vergleich.** Fehler unter akzeptierten Fällen, extern `clean`, Augmentation minus Kontrolle. Alles andere ist deskriptiv/explorativ, ohne Multiplizitätskorrektur.

**Nachtrag (nach dem Ziehen der Stichprobe, vor jeder Modellauswertung).** Die DICOM-Metadaten zeigen 131 Bilder mit Alter < 18 Jahren und gemischte Aufnahmerichtungen (AP/PA). Zusätzlich, rein explorativ: Kennzahlen für Alter ≥ 18 sowie getrennt nach AP und PA (AP ist in RSNA mit Verschattung assoziiert). Die Primärauswertung bleibt unverändert.

**Nachtrag (10. November 2026, nach der Primärauswertung).** Explorative Stratifikation nach DICOM-ViewPosition mit Bootstrap-Intervallen aus den bereits gespeicherten Vorhersagen (`experiments/external_stratify.py`, `reports/external/stratified_view.*`). Keine neuen Modelle, keine neuen Schwellen, keine Änderung der Primärauswertung. Die Abnahme der Kennzahlen beim Wechsel vom internen auf den externen Test betrifft einen Datensatz- und Populationswechsel (PneumoniaMNIST/Guangzhou, pädiatrisch → RSNA/NIH, überwiegend erwachsen; andere Aufnahme, Vorverarbeitung und Labeldefinitionen) und ist nicht nachgewiesen allein durch den Wechsel von Kindern zu Erwachsenen. Unterschiede zwischen AP und PA, auch in der AUROC innerhalb der jeweiligen Aufnahmeart, begründen eine Untersuchung möglicher Abhängigkeiten von der Aufnahmeart; sie beweisen keinen Aufnahmeart-Shortcut.
