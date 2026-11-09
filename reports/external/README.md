# Externer Test auf RSNA-Thoraxaufnahmen Erwachsener

Explorativ, Stand 9. November 2026. Protokoll vor der Auswertung festgelegt: [PROTOCOL.md](PROTOCOL.md). Eingefrorene 28px-CNNs aus Versuch 07 (Kontrolle vs. Rauschaugmentation, je drei Seeds) mit ihren unveränderten Annahmeschwellen (Ziel: 80 % Akzeptanz auf den Kalibrierungsdaten). Kein Training, keine neuen Schwellen.

**Daten.** 2.000 Bilder aus der RSNA Pneumonia Detection Challenge (Erwachsene, Labels von Radiologen-Panels; Bilder aus NIH ChestX-ray14): 1.000 `Lung Opacity` (positiv) und 1.000 `Normal` (negativ), mit festem Seed gezogen, eine Aufnahme pro Patient-ID ([Liste](../../configs/external_rsna_sample.csv)). Dazu getrennt 500 `No Lung Opacity / Not Normal`. Vorverarbeitung wie bei PneumoniaMNIST: Mittelzuschnitt, Bicubic auf 28×28.

| Clean, Mittel über 3 Seeds [95%-Bootstrap] | Kontrolle intern | Kontrolle extern | Augmentiert intern | Augmentiert extern |
|---|---:|---:|---:|---:|
| Accuracy | 84,1 % [81,3; 86,7] | 73,5 % [71,8; 75,2] | 84,1 % [81,3; 86,7] | 72,9 % [71,1; 74,7] |
| AUROC | 0,941 | 0,820 | 0,926 | 0,804 |
| ECE (15 Bins) | 0,109 | 0,141 | 0,122 | 0,133 |
| Annahmerate (feste Schwelle) | 78,5 % | 69,8 % | 82,2 % | 75,2 % |
| Fehler unter akzeptierten Fällen | 11,2 % [8,8; 13,8] | 20,6 % [18,7; 22,5] | 13,0 % [10,5; 15,9] | 21,8 % [19,8; 23,7] |

Primärer externer Vergleich (Fehler unter akzeptierten Fällen, clean, augmentiert minus Kontrolle): **+1,2 pp [+0,4; +1,9]**. Unter Rauschen (SD 0,1) extern: Kontrolle 45,6 %, augmentiert 22,5 % Fehler unter akzeptierten Fällen (Differenz −23,1 pp [−26,3; −19,9]); intern waren es 57,0 % vs. 12,6 %.

![Reliability Diagram](reliability.png)
![Risk-Coverage intern vs. extern](risk_coverage.png)

**Befund.** Der Domänenwechsel verdoppelt etwa den Fehler unter akzeptierten Fällen (≈11–13 % → ≈21–22 %). Die Schwelle lehnt extern zwar mehr ab (Annahme ≈70–75 % statt ≈80 %), fängt den Einbruch aber nicht auf. Bei 80 % Abdeckung ist das externe Risiko etwa doppelt so hoch wie das interne. Nur die Kontrollmodelle kommen bei sehr geringer Abdeckung (≤ 10 %) wieder in die Nähe des internen Niveaus (Risk-Coverage-Abbildung). Der Nutzen der Rauschaugmentation überträgt sich auf verrauschte externe Bilder, bringt auf sauberen externen Bildern aber keinen Vorteil. Von den 500 `Not Normal`-Fällen stufen die Modelle ≈64 % als Pneumonie ein; unter den akzeptierten davon sind ≈73–75 % falsch. Andere Auffälligkeiten werden also selbstsicher als Pneumonie gelesen. Alle Bedingungen: [conditions.png](conditions.png), [results.csv](results.csv), [summary.json](summary.json).

**Grenzen.** Kinder (Training) → Erwachsene (Test) und anderes Gerät, andere Aufnahmerichtung (AP/PA gemischt) und ein anderer Bildausschnitt: Der Effekt lässt sich nicht einer einzelnen Ursache zuordnen. `Lung Opacity` ist nur ein Stellvertreter für Pneumonie, und die Labels sind verrauscht. Die Auflösung beträgt 28×28. Die MedMNIST-Interpolation ist nicht dokumentiert (Bicubic angenommen). 131 Bilder stammen von Personen unter 18 Jahren; die Auswertung nur mit Erwachsenen ändert wenig (`external_adult` in `results.csv`). Die Aufnahmerichtung ist mit dem Label verknüpft: AP-Aufnahmen haben 79 % Prävalenz, PA-Aufnahmen 23 %. Die Accuracy liegt bei AP bei ≈81 %, bei PA bei ≈66 %. Das Modell könnte also teils Aufnahmemerkmale statt Befunde nutzen (`external_AP`/`external_PA`). Die Patient-IDs sind nicht mit NIH-Patienten verknüpft. Die Intervalle gelten bedingt auf feste Modelle und Schwellen. Die Bilder stammen aus einem inoffiziellen Hugging-Face-Spiegel (Revision fixiert, Prüfsummen geprüft). Es gibt keinen klinischen Anspruch.

**Daten und Lizenz.** RSNA Pneumonia Detection Challenge (2018), Shih et al., *Radiology: AI* 2019, [doi:10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041). Die Nutzung ist laut RSNA für nicht-kommerzielle Forschung und Lehre mit Quellenangabe vorgesehen. Die Bilder stammen aus NIH ChestX-ray14, Wang et al., CVPR 2017. Die Rohbilder werden nicht im Repository verteilt; nur die Patient-IDs und Prüfsummen liegen hier.

**Reproduzieren.** `python experiments/external_data.py` lädt etwa 340 MB, ohne Login. Danach `python experiments/external_eval.py` (etwa 1 Minute auf der CPU) und die Tests mit `python -m unittest tests.test_external`.
