# SwiftUI-Modelltest

Dieser Quelltext ist ein Einstieg für einen technischen iOS-Test, keine fertige Diagnose-App. Er wurde unter Linux erstellt; Xcode-Build, Apple-Laufzeit und Gerätegeschwindigkeit sind hier nicht geprüft.

1. Nach abgeschlossenem Training `python scripts/export_original_coreml.py` ausführen; optionale Pakete aus `requirements-coreml.txt`.
2. In einem SwiftUI-Projekt mit iOS 16+ `TensorProbeView.swift` hinzufügen und als View verwenden.
3. Die lokalen Artefakte aus `data/original_xray/deployment/` hinzufügen: `Mobile512Research.mlpackage`, `gradient512.f32`, `fixture.json`. Xcode kompiliert das Modell; die Referenzdateien müssen zum App-Bundle gehören.
4. Zuerst den synthetischen Tensorvergleich auf Apple-Hardware ausführen. Der Prüfschritt testet Modellrechnung, noch keine Bildvorverarbeitung oder klinische Qualität.

Der Modelleingang ist Float32 `[1,1,512,512]` mit Grauwerten 0–255. RGB-Wiederholung und ImageNet-Normalisierung stecken im exportierten Modell. Die bisherige Bildvorverarbeitung erhält das Seitenverhältnis, verwendet Pillow-Lanczos und schwarze Ränder. Eine spätere Implementierung mit Core Graphics/Vision muss Ausrichtung, Grauwertkonvertierung, Interpolation und Padding gegen identische Referenzbilder prüfen. Ein automatisches Vision-Cropping wäre ein anderer Eingabepfad.

Die Ausgabe `logit` gehört zur Klasse PNEUMONIA. `sigmoid(logit)` ist ein Modellsignal, keine nachgewiesen kalibrierte individuelle Erkrankungswahrscheinlichkeit. Die Annahmeschwelle aus der Forschungsstudie ist keine klinische Freigaberegel. Gewichte und Daten werden durch die MIT-Code-Lizenz nicht neu lizenziert; siehe [Datenherkunft](../../docs/LICENSING.md).

## Bildvorverarbeitung mit vier Referenzfällen prüfen

Die [512px-Stressanalyse](../../reports/original_512/preprocessing/README.md) zeigt: Allein bilineare statt Lanczos-Interpolation verändert 24/618 Entscheidungen. Deshalb genügt der erfolgreiche Tensorvergleich nicht für einen Bildimport.

`python scripts/make_preprocessing_fixtures.py` erzeugt einmalig das lokale Paket `data/original_xray/PreprocessingParityFixtures.zip`. Vier synthetische Fälle enthalten `source.png`, `expected512.png`, `gradient512.f32` und `fixture.json`. Die [versionierte Manifestdatei](../../reports/original_512/preprocessing/fixtures.json) enthält Geometrie, Referenzlogits und Prüfsummen.

1. Pro Fall die eigene Bildpipeline auf `source.png` anwenden und die Rohpixel mit `expected512.png` beziehungsweise dem Float32-Tensor vergleichen. Der Referenzvertrag verlangt exakte Pixelidentität; ein tolerierbarer klinischer Pixelunterschied wurde nicht ermittelt.
2. Pillow konvertiert zuerst nach Graustufen, skaliert das volle Bild mit Lanczos und erhält das Seitenverhältnis. Abmessungen werden zur nächsten geraden Zahl gerundet, wenn genau eine halbe Einheit vorliegt: 1024×641 ergibt 512×320, 1024×643 ergibt 512×322. Schwarzes Padding wird links/oben mit ganzzahlig abgerundetem halbem Rest angesetzt.
3. Für die separate Modellprüfung jeweils `gradient512.f32` und `fixture.json` eines Falls ins App-Bundle kopieren und den bestehenden Tensorprüfer starten. Die Logittoleranz von 0,001 ist eine technische Prüftoleranz, keine klinische Freigabe.

Die synthetischen Bilder sind bereits aufrecht ausgerichtet; EXIF, DICOM-Fensterung und Kamerabilder sind nicht abgedeckt. Das bisherige `SwiftUIResearchBundle.zip` bleibt unverändert. Ein Xcode-/Gerätetest steht weiter aus.
