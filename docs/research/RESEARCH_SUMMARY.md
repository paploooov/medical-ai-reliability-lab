# Arbeitsstand für eine spätere Facharbeit

**Stand: 9. November 2026. Forschungsnotiz, kein begutachteter Fachartikel.**

## Fragestellung

Wie verändert sich die Fehlererkennung eines Bildklassifikators, wenn sich die Qualität medizinischer Bilder verschlechtert? Dabei reicht es nicht, die Genauigkeit zu messen. Ein System kann falsche Vorhersagen mit hoher Sicherheit ausgeben; umgekehrt kann ein Warnsystem so viele richtige Fälle zurückhalten, dass sein Nutzen gering wird.

Wir untersuchen deshalb drei getrennte Eigenschaften: die Richtigkeit der Klassifikation, die Aussagekraft ihrer Wahrscheinlichkeiten und die Möglichkeit, anhand dieser Wahrscheinlichkeiten fehleranfällige Fälle zur Prüfung zurückzuhalten. Der Versuch ist auf einen öffentlichen pädiatrischen Röntgenbenchmark beschränkt. Er bewertet keine tatsächliche Versorgungssituation.

## Daten und methodische Entwicklung

Die ersten Versuche nutzten die offiziellen PneumoniaMNIST-Splits. Eine anschließende Datenprüfung fand acht unterschiedliche pixelidentische Bilder sowohl in Training als auch Validierung sowie Wiederholungen innerhalb der Splits. Deshalb wurden diese Ergebnisse als Vorversuche erhalten und eine gesonderte Analyse mit Gruppen pixelidentischer Bilder entwickelt. Auch die 64×64-Version enthält entsprechende Überschneidungszahlen.

Seit Versuch 04 werden 4.682 Trainingsfälle, 262 Fälle zur Modellwahl, 262 getrennte Kalibrierungsfälle und 621 eindeutige Testrepräsentanten verwendet. Ein Modell wird anhand des Auswahlverlusts bestimmt; erst danach wird anhand der Kalibrierungsfälle eine Konfidenzschwelle für ungefähr 80% Akzeptanz gewählt. Kein pixelidentischer Zusammenhang überschreitet diese Bereiche. Das bedeutet ausdrücklich nicht, dass unbekannte Patientenüberschneidungen ausgeschlossen sind.

Die Modelle sind kleine selbst trainierte CNNs. Es handelt sich um Standardarchitekturen mit projektspezifischen Gewichten, nicht um neu erfundene medizinische Foundation-Modelle. Drei Initialisierungen werden vollständig berichtet. Änderungen an der Methode sind im Versuchsverzeichnis nachvollziehbar; sie sind keine unabhängigen Replikationen auf neuen Fällen.

## Zentrale Befunde

**1. Konfidenz ersetzt keine Fehlerprüfung.** Schon die ersten Baselines blieben unter Störungen sicher, obwohl sie mehr Fehler machten. Eine einfache Unsicherheitsschwelle markierte nur einen Teil der Fehler und hielt zugleich richtige Vorhersagen zurück.

**2. Ein Auflösungsvorteil hängt von der Vergleichsregel ab.** Bei festen, auf sauberen Daten gewählten Schwellen stieg die Fehlerrate unter akzeptierten Fällen bei Unschärfe mit 64×64 weniger stark als mit 28×28. Beide Regeln akzeptierten jedoch unterschiedlich viele Fälle. Beim rückblickenden Vergleich von jeweils 80% Akzeptanz betrug die Differenz des Fehleranstiegs nur noch etwa 2,5 Prozentpunkte. Die beiden Analysen beantworten verschiedene Fragen; ihre Differenz ist keine kausale Zerlegung.

**3. Längeres Training allein behebt das Rauschproblem nicht.** Ein festgelegter Vergleich von 15 und 45 Epochen zeigte bei 28×28 weiterhin etwa 40% Genauigkeit unter dem getesteten Rauschen. Bei 64×64 sank die mittlere Rauschgenauigkeit von 76,1 auf 65,6%. Die ersten 15 Epochen der verlängerten Läufe stimmen mit dem ursprünglichen Verlauf überein.

**4. Ein interner Fehlmechanismus ist direkt überprüfbar.** Unter Rauschen ist bei zwei der früheren 28px-Modelle der gesamte letzte Merkmalsvektor für die meisten Fälle null. Die Ausgabe reduziert sich dann auf den Bias der letzten Schicht. Die Annahmeschwelle akzeptiert viele dieser konstanten Ausgaben weiterhin. Ein Nullvektor allein beweist aber weder eine Fehlentscheidung noch dauerhaft „tote“ Neuronen: Auch korrekte negative Fälle können auf diesem Weg klassifiziert werden.

**5. Eine gezielte Intervention hilft, aber nicht überall.** Frisches Gaußrauschen auf der Hälfte der Trainingsbilder erhöht beim gleichen 45-Epochen-Budget die mittlere Genauigkeit unter der entsprechenden Rauschbedingung von 40,0 auf 84,0%. Die mittlere Accuracy auf Originalbildern bleibt bei etwa 84,1%. Trotzdem steigt dort die Fehlerrate unter den von der jeweiligen Regel akzeptierten Fällen von 11,2 auf 13,0%; unter Kontrastverlust steigt sie ebenfalls leicht. Störungsspezifisches Training ist daher kein allgemeiner Sicherheitsnachweis.

## Statistische Aussagekraft

Die Berichte enthalten gepaarte Bootstrap-Intervalle: dieselben Fallindizes werden über verglichene Bedingungen und Modelle hinweg gemeinsam gezogen. Die aktuell verwendeten Einheiten sind eindeutige Pixelgruppen, keine nachgewiesenen unabhängigen Patienten. Die Intervalle halten Gewichte und Kalibrierungsschwellen fest. Sie enthalten weder die gesamte Trainingsvariabilität noch die Unsicherheit durch eine andere kleine Kalibrierungsstichprobe.

Minima und Maxima über drei Seeds sind als solche beschriftet und keine Konfidenzintervalle. Zusätzliche Rauschfolgen sind keine neuen Patienten. Die primären Endpunkte einzelner Folgeversuche wurden vor deren Berechnung festgelegt, die Themenwahl entstand aber nach Sichtung früherer Ergebnisse. Daher bleiben die Aussagen explorativ.

## Was sich noch nicht sagen lässt

Die Arbeit belegt weder klinischen Nutzen noch eine neue wissenschaftliche Methode. Rauschen, Unschärfe und Kontraständerungen sind synthetische Eingriffe; sie bilden keinen bestimmten Scanner und keinen Krankenhauswechsel verlässlich ab. Höhere MedMNIST-Auflösung liefert dieselben Fälle mit mehr Bildpunkten, keine externe Validierung. Die meisten Quellen stammen aus einem einzelnen pädiatrischen Benchmark, während potenzielle externe Quellen andere Populationen und Labeldefinitionen haben.

Die nächste belastbare Erweiterung muss über weiteres Optimieren dieses bekannten Testsatzes hinausgehen: eine begründete externe Zielpopulation und Labeldefinition, überprüfbare Patientengruppen, zulässiger Datenzugang und eine vorab eingefrorene Auswertung. Die geprüften Kandidaten und ihre Hürden stehen in [Externe Daten](EXTERNAL_DATA.md).

## Grundlage und Reproduzierbarkeit

Konfidenzbasierte Fehlererkennung, Temperature Scaling und selektive Vorhersagen sind etablierte Ansätze. Der [Literaturüberblick](LITERATURE.md) ordnet sie anhand von Hendrycks & Gimpel, Guo et al., Ovadia et al., Geifman & El-Yaniv sowie medizinischen Generalisierungsarbeiten ein. Er ist noch keine systematische Literaturübersicht.

Zu jedem Versuch gibt es Code, Konfiguration, Modellgewichte oder einen Verweis auf verwendete Gewichte, Vorhersagen, Diagramme und Grenzen. Softwaretests prüfen Berechnungen; zusätzliche Nachprüfungen laden Gewichte erneut und rechnen Ergebniszeilen nach. Die Trennung zwischen mathematisch korrekter Auswertung und medizinischer Aussagekraft bleibt wesentlich.

[Versuchsverzeichnis](EXPERIMENTS.md) · [Budgetvergleich](BUDGET_STUDY.md) · [Augmentationsversuch](AUGMENTATION_STUDY.md)
