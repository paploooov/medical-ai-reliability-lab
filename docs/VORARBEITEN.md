# THORAXIS — bisherige Teilstudien

Stand: 11. November 2026. Diese Übersicht bündelt die Arbeiten aus `reports/`, `docs/archive/` und `docs/research/`, einschließlich späterer technischer Reproduktionsprüfungen; Protokoll und Ergebnisbericht werden gemeinsam behandelt, reine Verweisdateien nicht doppelt gezählt. Alte Vollberichte sind über die Archivlinks in der Projekt-README erreichbar; hier führen Links direkt zu gespeicherten Daten.

Alle klinischen Befunde sind explorativ; wiederholt untersuchte Testfälle sind keine unabhängigen Bestätigungskohorten. „pp“ bedeutet Prozentpunkte, „akzeptiert“ die nach einer Konfidenzregel behaltenen Vorhersagen; Unsicherheitsintervalle gelten jeweils nur unter den im verlinkten Datensatz dokumentierten Annahmen.

## Frühe Modelle und Datenprüfung

### 01 · Pilot: lineares Modell und MLP

Zwei CPU-Modelle wurden auf 624 offiziellen PneumoniaMNIST-Testbildern und synthetischen Bildstörungen untersucht; beim MLP stieg der Anteil hochkonfidenter Fehler unter starkem Kontrastverlust um **14,7 pp [12,0; 17,5]**. Später gefundene Training-/Validierungsduplikate begrenzen diese historische Baseline; gespeichert sind [Kennzahlen](../reports/metrics/results.csv), [Vorhersagen](../reports/metrics/predictions.npz) und [Modellwahl](../reports/metrics/validation.json).

### 02 · Fehlererkennung aus Modellunsicherheit

Die Unsicherheit des eingefrorenen Pilot-MLP erreichte auf Originalbildern eine **Fehlererkennungs-AUROC von 0,736**; diese Kennzahl ist von der Pneumonie-AUROC zu unterscheiden. Die Validierungsschwelle markierte 54 % der Fehler, ließ aber viele Fehler passieren; [Kennzahlen und Schwellen](../reports/metrics/error_detection.json) sowie [vollständige Tabelle](../reports/metrics/error_detection.csv) sind gespeichert.

### 03 · Erstes CNN mit drei Trainingsseeds

Ein CNN mit 105.281 Parametern wurde mit drei Seeds und je 25 Epochen trainiert und erreichte auf Originalbildern im Mittel **84,8 % Accuracy**. Temperature Scaling verbesserte die Testwerte nicht durchgehend, und Modellwahl sowie Temperatur-Fit teilten damals denselben Validierungssatz; [Ergebnisse](../reports/cnn/results.csv), [Vorhersagen](../reports/cnn/predictions.npz) und [Lernkurven](../reports/cnn/learning_curves.csv) erhalten diesen frühen Stand.

### 04 · Daten- und Artefaktaudit

Der Audit fand **acht pixelidentische Training-/Validierungsüberschneidungen**, 18 überschüssige Trainingswiederholungen und drei Testwiederholungen, auch im offiziellen 64px-Archiv. Die Prüfung bestätigte rechnerische Konsistenz der frühen Ergebnisse, aber keine Patientenunabhängigkeit; [Artefaktprüfung](../reports/audit/artifact_checks.json), [64px-Audit](../reports/audit/resolution_64.json) und [Datensatzübersicht](../reports/dataset_summary.json) dokumentieren die Befunde.

## Kleine CNNs: Auflösung, Robustheit und Annahmeregeln

### 05 · Kontrollierter 28-/64px-Vergleich

Nach Pixel-Deduplizierung wurden sechs gleich große CNNs auf 621 Testrepräsentanten verglichen; der Unterschied des Unschärfe-minus-Original-Anstiegs akzeptierter Fehler betrug **−10,09 pp [−12,03; −8,23]**, 64 minus 28 Pixel. Die Akzeptanzmengen waren unterschiedlich und die Pixelgruppen keine bestätigten Patienten; [Primäranalyse](../reports/resolution/primary_analysis.json), [Vorhersagen](../reports/resolution/predictions.npz) und [Splitmanifest](../configs/resolution_manifest.json) sichern die Nachrechnung.

### 06 · Gleiche Abdeckung und Fehleratlas

Bei rückblickend gleicher Abdeckung von 80 % reduzierte sich der 64px-Vorteil beim Unschärfe-bedingten Fehleranstieg auf **−2,50 pp [−4,16; −0,72]**. Diese Nachanalyse zeigt den Einfluss unterschiedlicher Annahmemengen und behandelt Konfidenzgleichstände anteilig; [Primärergebnis](../reports/selective_review/primary.json), [Abdeckungstabelle](../reports/selective_review/matched_coverage.csv) und [Fehleratlas](../reports/selective_review/error_atlas.json) sind gespeichert.

### 07 · Nullvektoren und längeres Trainingsbudget

Bei zwei 28px-Modellen verschwanden unter Rauschen sämtliche letzten Merkmale für **597 beziehungsweise 589 von 621 Fällen**, sodass nur der Ausgabebias blieb. Eine kontrollierte Erweiterung von 15 auf 45 Epochen änderte den primären Fehleranstieg um lediglich −3,31 pp und löste das Problem nicht grundsätzlich; [Diagnostik](../reports/budget/noise_diagnostics_15.npz), [Primäranalyse](../reports/budget/primary.json) und [Vorhersagen](../reports/budget/predictions.npz) sind erhalten.

### 08 · Rauschstärken-Sweep der kleinen CNNs

Die eingefrorenen Modelle wurden bei **sechs Rauschstärken von SD 0 bis 0,2** mit drei Rauschrealisierungen auf Accuracy und Nullvektoranteil untersucht. Die Analyse ist eine explorative Mechanismusprüfung ohne erneutes Training; [Messraster](../reports/budget/noise_severity.csv) und [unabhängige Prüfung](../reports/budget/sweep_verification.json) speichern die vollständigen Verläufe.

### 09 · Gezielte Rauschaugmentation, drei Seeds

Bei gleichem 45-Epochen-Budget senkte Rauschaugmentation den Rauschen-minus-Original-Anstieg akzeptierter Fehler gegenüber der Kontrolle um **46,33 pp**. Der Effekt betrifft primär die trainierte Rauschfamilie und belegt keine allgemeine Robustheit; [Primäranalyse](../reports/augmentation/primary.json), [Armvergleich](../reports/augmentation/comparison.csv) und [gepaarte Vorhersagen](../reports/augmentation/comparison_predictions.npz) sind gespeichert.

### 10 · Unsicherheit der Kalibrierungsstichprobe

Gemeinsame Wiederziehungen von Kalibrierungs- und Testfällen ergaben für denselben Kontrast −46,33 pp mit empirischem Bereich [−52,47; −40,84], aber **54 von 2.000 Ziehungen waren wegen leerer Annahmemengen undefiniert**. Der Bereich ist auf definierte Ziehungen bedingt und keine sichere Fehlergarantie; [Zusammenfassung](../reports/calibration/summary.json), [Bootstrapdaten](../reports/calibration/bootstrap.npz) und [Teilmengenanalyse](../reports/calibration/subset_sensitivity.csv) enthalten auch die Sensitivität gegenüber kleineren Kalibrierungssätzen.

### 11 · Mechanismus der Konfidenzgleichstände

Alle **250 Modell-Fall-Paare** auf den fünf betroffenen Kalibrierungsschwellen hatten einen vollständig leeren letzten Merkmalsvektor und damit exakt den Bias-Logit. Deterministisches Tie-Breaking erreichte 209/262 akzeptierte Kalibrierungsfälle, während Nullvektor-Ausschluss teils die verbleibende Fehlerrate erhöhte oder alle Fälle entfernte; [Mechanismusdaten](../reports/mechanistic_ties/mechanism.npz), [Regelvergleich](../reports/mechanistic_ties/outcomes.csv) und [leere Bootstrapmengen](../reports/mechanistic_ties/bootstrap_empty.csv) dokumentieren dies.

### 12 · Transfer der Rauschaugmentation auf Blur und Kontrast

Die sechs eingefrorenen Modelle wurden unter anderen, bereits bekannten Störungen verglichen; bei augmentierten Modellen lag die akzeptierte Fehlerrate unter Kontrastverlust bei **20,08–30,89 %**. Wenige Nullvektoren bedeuteten somit keine zuverlässige Fehlererkennung, und ihr Ausschluss war kein genereller Sicherheitsgewinn; [Ergebnisse](../reports/transfer/results.csv), [Kalibrierungsregeln](../reports/transfer/calibration.csv) und [Vorhersagen](../reports/transfer/predictions.npz) sind erhalten.

### 13 · Erweiterung auf fünf gepaarte Trainingsseeds

Zwei zusätzliche Seed-Paare erweiterten den Augmentationsvergleich auf fünf Paare, mit einem mittleren primären Kontrast von **−43,67 pp** und approximativem Seed-t-Intervall [−56,71; −30,63]. Das separate Bild-Bootstrapintervall erfasst eine andere Unsicherheitsquelle und ist kein gemeinsames Trainings-/Patientenintervall; [Zusammenfassung](../reports/five_seed/analysis/summary.json), [Seed-Ergebnisse](../reports/five_seed/analysis/per_seed.csv) und [Kennzahlen](../reports/five_seed/analysis/results.csv) sind gespeichert.

### 14 · Temperature Scaling auf getrennten Kalibrierungsdaten

Bei sechs eingefrorenen Modellen verschlechterte der ausschließlich auf Kalibrierungsbildern angepasste Temperatur-Fit die Test-NLL in allen vier Bedingungen; auf Originalbildern stieg sie bei der Kontrolle von **0,505 auf 0,585**. ECE, Brier und NLL können dabei unterschiedliche Entwicklungen zeigen; [Ergebnisse](../reports/temperature_scaling/results.csv), [Zusammenfassung](../reports/temperature_scaling/summary.json) und [Logits](../reports/temperature_scaling/logits.npz) ermöglichen die Nachrechnung.

### 15 · Wiederholter Temperatur-Fit unter Resampling

Bei zusätzlicher Variation des Kalibrierungssatzes umfasste der empirische Bereich der NLL-Änderung für die Kontrolle **−0,022 bis +0,346** und damit null. Die 2.000 Wiederziehungen relativieren die Aussage des zuvor festgehaltenen Temperatur-Fits, enthalten aber keine erneute Modelloptimierung oder Patientencluster; [Zusammenfassung](../reports/temperature_scaling/sensitivity/summary.json) und [Bootstrapdaten](../reports/temperature_scaling/sensitivity/bootstrap.npz) sind gespeichert.

## Externe kleine Modelle und 224px-Studien

### 16 · Externer RSNA-Test

Auf einer festen externen Stichprobe mit 2.000 Bildern stieg der Fehler unter akzeptierten Vorhersagen der 28px-Kontrolle von intern 11,2 % auf **20,6 % [18,7; 22,5]**. Unterschiede in Population, Labels und Aufnahmeart sowie offene Weitergaberechte des verwendeten Spiegels begrenzen die Interpretation; [Ergebnisse](../reports/external/results.csv), [Zusammenfassung](../reports/external/summary.json) und [Vorhersagen](../reports/external/predictions.npz) bleiben erhalten. Zusätzlich wurden 500 Fälle der Kategorie „No Lung Opacity / Not Normal“ separat deskriptiv geprüft.

### 17 · AP/PA-Untergruppen und Klassenstandardisierung

Nach Sichtung der Aufnahmeart erreichte bereits die rein auf AP/PA beruhende Regel „AP positiv, PA negativ“ **78,4 % Accuracy** in der externen Stichprobe. Die anschließende gleiche Klassengewichtung und stratifizierte Wiederziehung beschreiben die starke Verknüpfung von Aufnahmeart und Label, beweisen aber nicht, welche Merkmale die CNNs verwenden; [Untergruppentabelle](../reports/external/view_summary.csv), [standardisierte Analyse](../reports/external/view_analysis/summary.json) und [Bootstrapdaten](../reports/external/view_analysis/bootstrap.npz) sind gespeichert.

### 18 · 224px-Information gegenüber reduzierter 28px-Information

Ein eingefrorener ResNet-18 mit jeweils trainiertem linearem Kopf erreichte mit 224px-Information AUROC **0,980 gegenüber 0,960**, Differenz +0,019 [+0,009; +0,032]. Dieser Vergleich auf dem offiziellen 224px-Archiv ist eine interne Informationsprüfung und kein klinischer Originalbildtest; [Zusammenfassung](../reports/high_resolution/summary.json), [Ergebnisse](../reports/high_resolution/results.csv) und [Vorhersagen/Köpfe](../reports/high_resolution/predictions_and_probes.npz) sind gespeichert.

### 19 · Robustheit der eingefrorenen Auflösungsmodelle

Eingangsrauschen erhöhte beim 224px-Modell den akzeptierten Fehleranteil von **4,2 % auf 35,6 %**, während die Akzeptanz von 73,3 % auf 96,8 % stieg. Vier Störungen und zwei Rauschpositionen wurden kontrolliert verglichen; der primäre Unterschied des Risikoanstiegs bei 80 % Abdeckung zwischen den Auflösungsarmen war nicht eindeutig, siehe [Zusammenfassung](../reports/high_resolution/robustness/summary.json), [Ergebnisse](../reports/high_resolution/robustness/results.csv) und [Vorhersagen](../reports/high_resolution/robustness/predictions.npz).

### 20 · Rauschaugmentation des linearen 224px-Kopfes

Bei eingefrorenem Encoder senkte das Training des Kopfes mit Rauschen den Fehler bei 80 % Abdeckung unter Testrauschen um **15,5 pp [11,8; 19,3]** gegenüber dem clean-Kopf. Clean-Nachteile und weitere Rauschrealisierungen wurden mitberichtet; [Zusammenfassung](../reports/high_resolution/augmentation/summary.json), [Ergebnisse](../reports/high_resolution/augmentation/results.csv) und [Vorhersagen](../reports/high_resolution/augmentation/predictions.npz) sind gespeichert.

### 21 · Kontrolle der Regularisierungsstärke

Eine nachträgliche Kontrolle verglich clean- und rauschtrainierte Köpfe bei gleichem C; bei **C = 0,01** sank der Rauschfehler bei 80 % Abdeckung von **22,7 % auf 8,0 %**. Damit blieb der Vorteil auch bei dieser angeglichenen Regularisierung sichtbar, ohne einen neuen Gewinner zu wählen; [Zusammenfassung](../reports/high_resolution/augmentation/regularization_sensitivity/summary.json) und [Vorhersagen/Köpfe](../reports/high_resolution/augmentation/regularization_sensitivity/predictions_and_heads.npz) dokumentieren alle vier Kombinationen.

### 22 · Störstärkenanalyse bei 224px

Bei doppelter Trainingsrauschstärke, SD 0,2, betrug die Risikodifferenz bei 80 % Abdeckung **−4,43 pp [−5,84; −2,62]**, Augmentation minus Kontrolle. Das Raster umfasst außerdem stärkere Unschärfe sowie ein gemeinsames empirisches Unsicherheitsband; [Zusammenfassung](../reports/high_resolution/severity/summary.json), [Rasterwerte](../reports/high_resolution/severity/results.csv) und [Vorhersagen](../reports/high_resolution/severity/predictions.npz) sind gespeichert.

### 23 · Gezielte Mischung aus Rauschen und Unschärfe

Eine nach dem beobachteten Blur-Problem entwickelte Mischaugmentation senkte bei Sigma 4 den Fehler bei 80 % Abdeckung gegenüber dem reinen Rauschkopf um **18,96 pp [15,94; 22,30]**. Vier Arme einschließlich einer Kontrolle mit weniger Rauschen zeigen auch Nebenwirkungen; die Intervention bleibt explorativ, siehe [Zusammenfassung](../reports/high_resolution/mixed_augmentation/summary.json), [Ergebnisse](../reports/high_resolution/mixed_augmentation/results.csv) und [Vorhersagen](../reports/high_resolution/mixed_augmentation/predictions.npz).

## Originalbilder und THORAXIS

### 24 · Originalbild-Audit und neue Aufteilung

Aus dem offiziellen Mendeley-Archiv verblieben nach 32 exakten Duplikatausschlüssen und 453 potenziellen Überschneidungen von Dateinamen-Gruppen **5.371 Bilder**, darunter 618 Testbilder. Die Gruppen sind keine unabhängig bestätigten Patientenidentitäten; [Manifest mit Ausschlüssen](../reports/original_512/manifest.json), [Quellenaudit](../reports/original_512/source_audit.json) und [Protokoll](../configs/original_512.json) halten Datenbasis und Verarbeitung fest.

### 25 · Originalbilder bei 224/512px und partielles Nachtraining

Der eingefrorene 512px-Arm zeigte gegenüber 224px keinen eindeutigen AUROC-Gewinn (+0,0096 [−0,0042; +0,0251]); das partiell nachtrainierte **THORAXIS 512-R1 erreichte AUROC 0,9669 und 82,7 % Accuracy**. Nachtraining senkte den Fehler bei 80 % Abdeckung gegenüber dem eingefrorenen 512px-Kopf um 2,63 pp, bei weiterhin nur 54,5 % Spezifität; [Zusammenfassung](../reports/original_512/summary.json), [Ergebnisse](../reports/original_512/results.csv) und [Vorhersagen](../reports/original_512/predictions.npz) sind gespeichert.

### 26 · Optimierer-angepasste Kopfkontrolle

Eine reine Kopf-Nachtrainingskontrolle verwendete dieselbe Initialisierung, AdamW-Einstellungen, Reihenfolge und drei Epochen wie das partielle Encodertraining. Der Fehler bei 80 % Abdeckung war beim partiellen Encodertraining **2,43 pp niedriger**, während die AUROC-Differenz nicht eindeutig war; [Zusammenfassung](../reports/original_512/head_control/summary.json) und [Kopf/Vorhersagen](../reports/original_512/head_control/head_and_predictions.npz) enthalten die Kontrolle.

### 27 · Größerer eingefrorener Encoder

ResNet-50 mit trainiertem linearem Kopf erreichte auf derselben Originalbild-Kohorte **AUROC 0,9487**, gegenüber 0,9528 für ResNet-18. Der größere Encoder zeigte damit keinen eindeutigen Gewinn, wobei Architektur und Vortraining gemeinsam variieren; [Zusammenfassung](../reports/original_512/large_encoder/summary.json), [Modellwahl](../reports/original_512/large_encoder/selection.json) und [Kopf/Vorhersagen](../reports/original_512/large_encoder/head_and_predictions.npz) sind gespeichert.

### 28 · Übertragung diagnostischer Schwellen

Eine ausschließlich auf Kalibrierungsdaten gesetzte Schwelle mit Ziel 95 % Spezifität erreichte bei THORAXIS 512-R1 im Test nur **52,4 % Spezifität**. Diese Nachanalyse verändert weder die bisherigen Gewichte noch den Export und belegt die Kalibrierungs-/Testlücke; [Schwellenanalyse](../reports/original_512/threshold_transfer/summary.json) und [Bootstrapdaten](../reports/original_512/threshold_transfer/bootstrap.npz) sind gespeichert.

### 29 · Klassenabhängiges Herkunftsaudit

Bei NORMAL-Bildern unterschieden bereits Originalbreite und -höhe Entwicklungs- und Testbestand mit AUROC 0,794; zusammen mit Erscheinungsmerkmalen waren es **0,849 [0,822; 0,875]**. Diese gruppierte Out-of-fold-Analyse misst Herkunft statt Erkrankung und erklärt Fehler nicht kausal; [Zusammenfassung](../reports/original_512/source_audit/summary.json), [Merkmale/Vorhersagen](../reports/original_512/source_audit/features_and_oof.npz) und [Fold-Modelle](../reports/original_512/source_audit/fold_models.npz) sichern die Nachrechnung.

### 30 · Vorverarbeitung des eingefrorenen 512px-Modells

Sechs Bedingungen wurden auf denselben Bildern verglichen: Center Crop änderte **108/618 Entscheidungen, 17,5 % [14,3; 20,7]**, und senkte Sensitivität von 99,5 % auf 89,4 %; bilinear statt Lanczos änderte 24 Entscheidungen. Das zeigt technische Eingriffseffekte, rechtfertigt aber keine Auswahl der nachträglichen Testgewinner-Pipeline; [Ergebnisse](../reports/original_512/preprocessing/results.csv), [gepaarte Intervalle](../reports/original_512/preprocessing/summary.json) und [Vorhersagen](../reports/original_512/preprocessing/predictions.npz) sind gespeichert.

### 31 · Externer ChestMNIST224-Transfer

THORAXIS 224-Linear erreichte auf einer vorab festgelegten NIH-abgeleiteten Stichprobe **AUROC 0,673 [0,636; 0,710]** gegen No-finding- und 0,548 gegen andere Krankheitslabels. Textlabels, fehlende Patientenverknüpfung und mögliche NIH-/RSNA-Überlappung begrenzen diese Auswertung, die THORAXIS 512-R1 nicht extern validiert; [Stichprobe](../reports/external_chest224/sampling.json), [Zusammenfassung](../reports/external_chest224/summary.json) und [Vorhersagen](../reports/external_chest224/predictions.npz) sind gespeichert.

### 32 · Externe Fehlerquote über die Abdeckung

Eine beschreibende Nachanalyse derselben externen Vorhersagen untersuchte **20 Abdeckungsstufen je Kontrollgruppe** ohne neue Schwellenwahl. Die feste interne Schwelle akzeptierte gegen No finding 38,8 % der Fälle mit 52,1 % Fehlern gegenüber 49,1 % ohne Ablehnung; [Kurvendaten](../reports/external_chest224/selective_coverage.csv), [Referenzkennzahlen](../reports/external_chest224/results.csv) und [Prüfprotokoll](../reports/external_chest224/selective_coverage_verification.json) dokumentieren den Befund.

## Technische Ergänzungen und Zusammenführungen

### 33 · Core ML und synthetische Bildreferenzen

THORAXIS 512-R1 wurde technisch nach Core ML konvertiert; **vier synthetische Bild-/Tensorfälle** prüfen zusätzlich Rundung, Graustufen, Padding und Skalierung. Python-Referenzen und Paketprüfsummen sind geprüft, Apple-Laufzeit und Xcode-Build bleiben offen; [Exportvertrag](../reports/original_512/deployment.json), [Referenzmanifest](../reports/original_512/preprocessing/fixtures.json) und [Pixelprüfung](../reports/original_512/preprocessing/fixture_verification.json) speichern die Nachweise.

### 34 · Ein-Befehl-Reproduktion des Hauptmodells

Ein vollständiger Wiederholungslauf mit einem Seed, frischer Python-Umgebung und neu berechneten Merkmalen benötigte **15,17 Minuten und 1.185,7 MiB Peak-RSS**, ohne Testauswertung; das hash-geprüfte Roharchiv wurde lokal bereitgestellt. Zehn vergleichbare serielle Läufe werden auf 2,53 Stunden geschätzt, während vollständiger Archiv-Neutransfer und Gewichte-Weitergaberechte offen bleiben; [Laufdaten](reproduction-512/training_run.json), [Ressourcen/Prüfungen](reproduction-512/stage_checks.json), [Gewichtsvergleich](reproduction-512/checkpoint_comparison.json) und [lokales Release-Manifest](reproduction-512/release_manifest.json) sind gespeichert.

### 35 · Übersichtsabbildungen und zusammenführende Forschungsberichte

Die Übersichtsfiguren und der **sechsseitige Follow-up-Bericht** bündeln bestehende Befunde, insbesondere Nullvektor-Effekte, fünf Seeds, Kalibrierung und Auflösungsstudien; sie sind keine zusätzlichen unabhängigen Experimente. Auch die späteren drei- und zweiseitigen Forschungsnotizen fassen vorhandene Ergebnisse zusammen; [Übersichtsdaten](../reports/figures/overview_summary.json), [Abbildungsherkunft](../reports/figures/overview_provenance.json) und [Follow-up-Prüfung](../reports/followup/verification.json) dokumentieren ihre Grundlage.

## Einordnung der übrigen Archivdateien

`docs/research/` enthält fünf reine Weiterleitungen zu bereits oben erfassten historischen Studien. Die Dateien `*_PROTOCOL.md`, `PROTOCOL.md`, `CNN_PROTOCOL.md` sowie `NEXT_STUDY.md` dokumentieren Versuchspläne und zählen ohne eigene Resultate nicht als zusätzliche Ergebnisstudien; `PROJECT_SCOPE.md`, `LITERATURE.md`, `MODELS.md`, `EXPERIMENTS.md`, `EXTERNAL_DATA.md` und `RESEARCH_SUMMARY.md` liefern Rahmen, Quellen oder übergreifende Zusammenfassungen. Einträge zu Prüfungen, Quellenzugang und Export sind ausdrücklich technische Vorarbeiten, keine neuen klinischen Wirksamkeitsnachweise.

Sämtliche ursprünglichen Berichte, Konfigurationen, Modelle, Vorhersagen, Prüfsummen und Reproduktionsskripte bleiben an ihren bisherigen Pfaden erhalten; diese Zusammenfassung ersetzt keine Rohdaten und erzeugt keine neuen Kennzahlen.
