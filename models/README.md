# Gespeicherte Modelle

Alle Modelle wurden für dieses Projekt auf PneumoniaMNIST trainiert. Es sind kleine Forschungsbaselines, keine medizinischen Foundation-Modelle und keine klinisch validierten Diagnosesysteme. Die Architektur stammt aus etablierten Standardbausteinen.

| Dateien | Architektur | Trainings-/Auswahlregel | Bericht |
|---|---|---|---|
| `logistic.npz`, `mlp.npz` | Logistische Regression; 784→64 ReLU→1 Sigmoid | Offizielle Splits, früher Pilot | [Versuch 01](../docs/research/PILOT.md) |
| `cnn/seed_*.pt` | Zwei Faltungsblöcke, 105.281 Parameter | Drei Seeds, 25 Epochen; offizielle Splits | [Versuch 03](../docs/research/CNN_STUDY.md) |
| `resolution/r*_s*.pt` | Zwei Faltungsblöcke, adaptives Pooling, 21.249 Parameter | Duplikatgruppen getrennt; 15 Epochen Budget, Auswahl nach Modellwahl-BCE | [Versuch 04](../docs/research/RESOLUTION_STUDY.md) |

Versuch 05 verwendet die Modelle von Versuch 04 unverändert. Die gewählte Auflösung ist Teil der Eingabespezifikation; eine adaptive Architektur allein macht die Gewichte nicht beliebig zwischen Auflösungen austauschbar.

## Eingabe und Ausgabe

Graustufenbilder, gemäß der jeweiligen Studie 28×28 oder 64×64, auf [0,1] skaliert. Keine zusätzliche Standardisierung. CNNs erwarten Form `[batch, 1, height, width]`; die ersten beiden Baselines erwarten zeilenweise abgeflachte Pixel. Die Ausgabe nach Sigmoid ist P(Datensatzklasse pneumonia). Entscheidungsschwelle: 0,5. Eine hohe Ausgabe ist kein Nachweis einer korrekten Diagnose.

PyTorch-Dateien enthalten State Dictionaries; mit `torch.load(path, weights_only=True)` und der passenden dokumentierten Modellklasse laden. Die NPZ-Dateien enthalten numerische Gewichte; keine Pickle-Dateien. Zustand `eval()` setzen. Hashes, Auswahl-Epochen und Schwellen stehen im `run.json` des jeweiligen Berichtsordners.

## Grenzen

Erhebliche Fehler unter künstlichen Störungen, nur ein bereits untersuchter pädiatrischer Datensatz und keine nachgewiesene Patientenunabhängigkeit. Für manche Modelle kollabiert die Entscheidung unter Störungen auf eine einzige Klasse. Die historische Modellleistung darf nicht als Geräte-, Krankenhaus- oder Patientenvalidierung dargestellt werden.

Ein Core-ML-Export und eine SwiftUI-App wurden noch nicht umgesetzt. Vor einem Export müssen Vorverarbeitung und numerische Gleichwertigkeit geprüft werden.
