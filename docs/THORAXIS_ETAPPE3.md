# THORAXIS: Herkunft, Korrelation und Interpolation

Abgeschlossen am 14. November 2026 · Protokoll `c44117a` unverändert. Code vor Ausführung veröffentlicht: **`5622c80`**, Herkunfts-Folds vor Training: **`0bba074`**. Keine medizinischen Modelle nachtrainiert, keine Schwellen geändert.

## Beobachtung

**Herkunfts-CNN:** Fünf gruppierte Außenfolds mit innerer Modellwahl liefern für 1.579 Normalbilder aus 1.443 Dateinamen-Gruppen eine Out-of-fold-AUROC von **0,869 [0,844–0,891]**. Das 95%-Intervall stammt aus 2.000 Gruppen-Bootstraps fester Vorhersagen. Die Balanced Accuracy bei 0,5 beträgt **0,682**. Die unabhängige paarweise Nachrechnung bestätigt die AUROC exakt. Der [frühere Wert](../reports/original_512/source_audit/summary.json) mit einfachen Bildmerkmalen war **0,849**; andere Folds und Verfahren erlauben daraus keinen gepaarten Überlegenheitsnachweis.

**Korrelation:** Zwischen OOF-Herkunftsscore und Pneumonie-Logit liegt das mittlere Spearman-ρ offiziell bei **0,239 [0,105–0,366]**, gemischt bei **0,575 [0,524–0,619]**. Innerhalb der ursprünglichen Herkunft bleibt es gemischt positiv: früheres Train **0,518 [0,456–0,573]**, früherer Test **0,201 [0,011–0,387]**. Angegeben sind Modellmittel und getrennte bedingte 95%-Gruppen-Bootstrap-Bereiche, keine künstlich vergrößerte Fallzahl.

Die alten Test-Normalbilder im gemischten Arm haben tatsächlich höhere mittlere Herkunftsscores: **0,503 statt 0,421**. Das passt zur vorherigen Spezifität **72,4 % statt 95,7 %**. Innerhalb einzelner Herkunfts-Außenfolds sind die alten Test-Untergruppen jedoch klein (**3–11 Bilder**) und die Korrelationen teils negativ; der Gesamtmittelwert verdeckt diese Unsicherheit.

**H3 — bilinear minus Lanczos**, unveränderte Modelle und eigene Testbilder:

| Arm | Änderung R80, Prozentpunkte [95%-Bootstrap] | Änderung Spezifität, Prozentpunkte | Entscheidungswechsel je 618 Bilder |
|---|---:|---:|---:|
| Offiziell | **+4,49 [3,01; 6,25]** | **−10,22** | 23,6 im Mittel; 22–26 |
| Gemischt | **+0,11 [−0,04; 0,31]** | **−1,54** | 7,64 im Mittel; 3–13 |

Die R80-Änderung im gemischten Arm bleibt **unklar**. Bei fester Ablehnung steigt der Fehler unter angenommenen Fällen offiziell von **6,75 % auf 10,71 %**, gemischt ändert er sich kaum (**0,554 % auf 0,526 %**). Die Annahmequoten ändern sich von **68,54 % auf 70,03 %** beziehungsweise **78,57 % auf 76,89 %**.

## Mögliche Erklärung

Die Bildbestände sind unterscheidbar, und ähnliche Bildeigenschaften könnten beide Modelle beeinflussen. Zusätzlich reagieren die medizinischen Modelle auf den Resize-Kernel. Herkunftsunterschiede und Vorverarbeitung sind damit plausible Ansatzpunkte für weitere Forschung.

**Alternativerklärungen (nachträglich explorativ):** Die [Zusatzanalyse](../configs/thoraxis_posthoc_partial_spearman.json) wurde nach Sichtung der Etappe-3-Ergebnisse festgelegt und gehört nicht zum Protokoll. Kontrolliert werden die Ränge von log Originalbreite/-höhe. Das mittlere partielle ρ beträgt offiziell **0,440**, gemischt **0,609**, innerhalb früherer Train-/Testherkunft **0,553/0,442**. Der Zusammenhang bleibt nach dieser begrenzten Größenkontrolle bestehen. Altersunterschiede der Kinder sind nicht prüfbar, da Altersdaten fehlen; Bildgröße ersetzt keine Altersmessung. Mögliche Grenzfälle in Normal-Labels sind ebenfalls ungeprüft. [Einzelwerte, Gruppen-Bootstrap und Nachrechnung](../reports/original_512/thoraxis_30/partial_spearman_dimensions.json).

## Fehlender Nachweis

Keine kausale Nutzung der Herkunft, kein anatomischer Shortcut und keine klinische Sicherheit nachgewiesen. Quelle und medizinisches CNN können Trainingsgruppen teilen; Herkunftsscores verschiedener Folds sind nicht zwingend gleich kalibriert. Dateinamen-Gruppen ersetzen keine bestätigten Patienten. Alle fünf Herkunfts-CNNs wählten Epoche 3, also die Budgetgrenze. Bootstrap-Bereiche gelten für feste Fits; Seed- und Split-Zusammenfassungen stehen separat in den Daten. H3 bleibt **explorativ**, da der Effekt auf diesem Testbestand vorher bekannt war; keine neue unabhängige Bestätigung. Null beobachtete Fehler sind kein Nullrisiko.

[Grafik 1: Herkunft/Korrelation](../reports/original_512/thoraxis_30/stage3_source_correlation.svg) · [Grafik 2: H3](../reports/original_512/thoraxis_30/stage3_h3.svg)

[OOF-AUROC/Folds](../reports/original_512/thoraxis_30/source_cnn_results.json) · [alle 30 Korrelationen samt Herkunft/Außenfolds](../reports/original_512/thoraxis_30/source_logit_correlations.json) · [Herkunftsscores/Fallzahlen](../reports/original_512/thoraxis_30/source_scores_by_origin.json) · [H3: Einzelwerte, Streuung, Bootstrap](../reports/original_512/thoraxis_30/h3_results.json). Unabhängige Prüfung: `python scripts/check_thoraxis_source_auc.py` ([Ergebnis](../reports/original_512/thoraxis_30/source_auc_independent_check.json)). Die fünf Herkunfts-Checkpoints bleiben lokal; [Pfade und Hashes](../reports/original_512/thoraxis_30/source_cnn_runs.json) sind gespeichert.
