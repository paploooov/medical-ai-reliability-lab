# SwiftUI-Modelltest

Dieser Quelltext ist ein Einstieg für einen technischen iOS-Test, keine fertige Diagnose-App. Er wurde unter Linux erstellt; Xcode-Build, Apple-Laufzeit und Gerätegeschwindigkeit sind hier nicht geprüft.

1. Nach abgeschlossenem Training `python scripts/export_original_coreml.py` ausführen; optionale Pakete aus `requirements-coreml.txt`.
2. In einem SwiftUI-Projekt mit iOS 16+ `TensorProbeView.swift` hinzufügen und als View verwenden.
3. Die lokalen Artefakte aus `data/original_xray/deployment/` hinzufügen: `Mobile512Research.mlpackage`, `gradient512.f32`, `fixture.json`. Xcode kompiliert das Modell; die Referenzdateien müssen zum App-Bundle gehören.
4. Zuerst den synthetischen Tensorvergleich auf Apple-Hardware ausführen. Der Prüfschritt testet Modellrechnung, noch keine Bildvorverarbeitung oder klinische Qualität.

Der Modelleingang ist Float32 `[1,1,512,512]` mit Grauwerten 0–255. RGB-Wiederholung und ImageNet-Normalisierung stecken im exportierten Modell. Die bisherige Bildvorverarbeitung erhält das Seitenverhältnis, verwendet Pillow-Lanczos und schwarze Ränder. Eine spätere Implementierung mit Core Graphics/Vision muss Ausrichtung, Grauwertkonvertierung, Interpolation und Padding gegen identische Referenzbilder prüfen. Ein automatisches Vision-Cropping wäre ein anderer Eingabepfad.

Die Ausgabe `logit` gehört zur Klasse PNEUMONIA. `sigmoid(logit)` ist ein Modellsignal, keine nachgewiesen kalibrierte individuelle Erkrankungswahrscheinlichkeit. Die Annahmeschwelle aus der Forschungsstudie ist keine klinische Freigaberegel. Gewichte und Daten werden durch die MIT-Code-Lizenz nicht neu lizenziert; siehe [Datenherkunft](../../docs/LICENSING.md).
