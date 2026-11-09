# Abschlusssynthese der Versuchsreihe 01–10

**Abgeschlossene explorative Studie auf PneumoniaMNIST, Forschungsstand v1.0.0. Kein begutachteter Fachartikel.**

Die Versuchsreihe endet mit Versuch 10. Dieser Abschluss ergänzt ausschließlich die redaktionelle Synthese und wiederholt bestehende Prüfungen; er enthält kein neues Experiment und kein neues Training.

## Gesamterkenntnis und mechanistischer Zusammenhang

**Ein konstantes, hochkonfidentes Modell kann häufig falsch sein; ein nichtkonstantes Modell ist deshalb noch nicht zuverlässig.** Die zehn Versuche verbinden die anfängliche Beobachtung hochkonfidenter Fehler mit einem exakt nachgewiesenen internen Mechanismus und dessen Grenzen.

Versuche 01–03 etablierten trainierte Baselines und zeigten, dass Accuracy, Wahrscheinlichkeitskalibrierung und Fehlererkennung verschiedene Eigenschaften sind. Der Daten-Audit führte in Versuch 04 zu Pixelgruppen-Deduplizierung und getrennten Modellwahl-/Kalibrierungssplits; Versuch 05 zeigte, wie stark Schlussfolgerungen von der Akzeptanzregel abhängen. Erst auf dieser eingegrenzten Grundlage folgten die mechanistische Diagnose und Gegenproben:

**Versuche 06 → 08 → 09: Nullvektor, Bias-Logit und Annahmeversagen.** Für den letzten Klassifikator gilt `z = wᵀh + b`. Bei vollständig nullgesetztem ReLU-Merkmalsvektor ist daher exakt `z = b`, also `p = sigmoid(b)`. Verschiedene Bilder erhalten denselben Logit und dieselbe Konfidenz. Alle 250 Modell-Fall-Paare exakt auf den fünf betroffenen Original-Kalibrierungsschwellen stammen aus dieser Nullvektorgruppe; Float-Sättigung und diskrete Wahrscheinlichkeitskalibrierung erklären diese Gleichstände nicht. Die Regel `confidence >= tau` akzeptiert die ganze Punktmasse. Dadurch verfehlt sie bereits auf der Kalibrierungsmenge das 80%-Ziel und nimmt unter Rauschen viele falsche konstante Ausgaben an. In 54 Kalibrierungs-/Testwiederziehungen liegt die Schwelle über allen Rauschkonfidenzen; die Annahmemenge ist leer. Tie-Breaking behandelt die Punktmasse, stellt verlorene Merkmalsunterschiede aber nicht wieder her.

Dies ist eine mathematisch und anhand gespeicherter Aktivierungen belegte Kausalkette innerhalb der festgehaltenen Modelle. Sie erklärt den Mechanismus dieser Gleichstände und dieses Versagens der Annahmeregel. Sie ist keine vollständige kausale Erklärung des Trainingsverlaufs, aller Rauschfehler oder sämtlicher Kalibrierungsverfahren.

**Versuche 07 → 10: Die Umkehrung gilt nicht.** Rauschaugmentation reduziert den massiven Kollaps unter der trainierten Rauschfamilie, verhindert Nullvektoren jedoch nicht vollständig. Unter Blur erreichen augmentierte Modelle trotz mehr Nullvektoren eine höhere Accuracy als ihre Kontrollen. Unter Kontrastreduktion haben sie nur 10, 3 beziehungsweise 0 Nullvektoren, aber 20,08 %, 22,09 % beziehungsweise 30,89 % Fehler unter den nach Standardregel akzeptierten Fällen. `h != 0` ist somit keine hinreichende Bedingung für Zuverlässigkeit. Die Störungen fehlten im Augmentationstraining, waren aber bereits bekannte Projekt-Testbedingungen; dies ist kein unabhängiger externer OOD-Test.

**Warum Nullvektor-Ausschluss auch richtige Vorhersagen entfernt.** Der Bias-only-Klassifikator gibt innerhalb dieser Aktivierungsregion stets dasselbe Label aus. Stimmt das Ziel eines Falls mit diesem Label überein, zählt die Vorhersage als korrekt, obwohl der letzte Merkmalsvektor keine Unterschiede zwischen diesen Fällen enthält. Ein Ausschluss anhand von `h = 0` unterscheidet solche korrekten von inkorrekten konstanten Vorhersagen nicht. Unter Blur und Kontrast kann er daher überwiegend korrekte Fälle entfernen und den verbleibenden Fehleranteil erhöhen.

Der Ausdruck „Mehrheitsklassen-Bias“ wäre hier sachlich falsch: Die [gespeicherten Bias-Werte](../../reports/transfer/calibration.csv) der fünf Modelle mit Schwellen-Gleichständen sind negativ; bei positivem Label „Pneumonie“ ergibt sich die konstante Vorhersage **„normal“**. Im [offiziellen Datensatz](../../reports/dataset_summary.json) überwiegt dagegen Pneumonie. Ein gemeinsam mit Gewichten gelernter Intercept ist nicht gleich dem Logit der Klassenhäufigkeit. „Richtiges Label ohne Bildinformation“ ist daher nur als Kurzform für fehlende **zwischen diesen Fällen variierende Information am letzten Merkmalsvektor** zulässig. Der Eingang beeinflusst weiterhin, ob das Netz in diese Nullaktivierungsregion gelangt; völlige Bildunabhängigkeit des gesamten Netzes wird nicht behauptet.

Der abgeschlossene Befund ist deshalb enger als ein allgemeines Sicherheitsverfahren: Nullaktivierung erklärt eine konkrete Konfidenz-Punktmasse; weder ihre Beseitigung noch ihr Herausfiltern garantiert brauchbare selektive Vorhersagen. [Abschlussprüfung und unveränderte Artefakte](../releases/v1.0.0/README.md).

## Fragestellung

Wie verändert sich die Fehlererkennung eines Bildklassifikators, wenn sich die Qualität medizinischer Bilder verschlechtert? Dabei reicht es nicht, die Genauigkeit zu messen. Ein System kann falsche Vorhersagen mit hoher Sicherheit ausgeben; umgekehrt kann ein Warnsystem so viele richtige Fälle zurückhalten, dass sein Nutzen gering wird.

Wir untersuchen deshalb drei getrennte Eigenschaften: die Richtigkeit der Klassifikation, die Aussagekraft ihrer Wahrscheinlichkeiten und die Möglichkeit, anhand dieser Wahrscheinlichkeiten fehleranfällige Fälle zur Prüfung zurückzuhalten. Der Versuch ist auf einen öffentlichen pädiatrischen Röntgenbenchmark beschränkt. Bei 28×28 Pixeln gehen radiologische Feinstrukturen verloren. Synthetisches Pixelrauschen dient als kontrolliertes ML-Testbed für niedriges Signal-Rausch-Verhältnis (SNR), nicht als physikalisches Modell klinischer Röntgendegradation. Die Untersuchung bewertet keine tatsächliche Versorgungssituation.

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

**6. Die Annahmeschwelle trifft ihr Ziel nicht zwingend.** Versuch 08 zieht die Kalibrierungsstichprobe erneut. Bei fünf Modellen liegen viele Fälle exakt auf derselben Konfidenz; die inklusive Annahmeregel akzeptiert deshalb auf dem Kalibrierungssatz 90–94% statt ungefähr 80%. In 54 von 2.000 gemeinsamen Wiederziehungen bleiben bei einem Kontrollmodell unter Rauschen keine akzeptierten Fälle: Die Fehlerrate ist dann undefiniert, nicht null. Der Augmentationsvorteil bleibt unter den auswertbaren Ziehungen bestehen; die Regel liefert dennoch keine Sicherheitsgarantie.

## Statistische Aussagekraft

Die Berichte enthalten gepaarte Bootstrap-Intervalle: dieselben Fallindizes werden über verglichene Bedingungen und Modelle hinweg gemeinsam gezogen. Die aktuell verwendeten Einheiten sind eindeutige Pixelgruppen, keine nachgewiesenen unabhängigen Patienten. Die früheren Intervalle halten Gewichte und Kalibrierungsschwellen fest; Versuch 08 variiert zusätzlich die Kalibrierungsfälle. Auch Versuch 08 erfasst nur empirische Kalibrierungsvariation innerhalb des vorhandenen Pools, weder die gesamte Trainingsvariabilität noch einen Wechsel der Patientenpopulation.

Minima und Maxima über drei Seeds sind als solche beschriftet und keine Konfidenzintervalle. Zusätzliche Rauschfolgen sind keine neuen Patienten. Die primären Endpunkte einzelner Folgeversuche wurden vor deren Berechnung festgelegt, die Themenwahl entstand aber nach Sichtung früherer Ergebnisse. Daher bleiben die Aussagen explorativ.

## Was sich noch nicht sagen lässt

Die Arbeit belegt weder klinischen Nutzen noch eine neue wissenschaftliche Methode. Rauschen, Unschärfe und Kontraständerungen sind synthetische Eingriffe; sie bilden keinen bestimmten Scanner und keinen Krankenhauswechsel verlässlich ab. Höhere MedMNIST-Auflösung liefert dieselben Fälle mit mehr Bildpunkten, keine externe Validierung. Die meisten Quellen stammen aus einem einzelnen pädiatrischen Benchmark, während potenzielle externe Quellen andere Populationen und Labeldefinitionen haben.

Eine mögliche eigenständige Folgestudie müsste über weiteres Optimieren dieses bekannten Testsatzes hinausgehen: eine begründete externe Zielpopulation und Labeldefinition, überprüfbare Patientengruppen, zulässiger Datenzugang und eine vorab eingefrorene Auswertung. Die geprüften Kandidaten und ihre Hürden stehen in [Externe Daten](EXTERNAL_DATA.md). Das ist ein Ausblick außerhalb der abgeschlossenen Versuchsreihe 01–10, kein noch ausstehender Versuch dieser Studie.

## Grundlage und Reproduzierbarkeit

Konfidenzbasierte Fehlererkennung, Temperature Scaling und selektive Vorhersagen sind etablierte Ansätze. Der [Literaturüberblick](LITERATURE.md) ordnet sie anhand von Hendrycks & Gimpel, Guo et al., Ovadia et al., Geifman & El-Yaniv sowie medizinischen Generalisierungsarbeiten ein. Er ist noch keine systematische Literaturübersicht.

Zu jedem Versuch gibt es Code, Konfiguration, Modellgewichte oder einen Verweis auf verwendete Gewichte, Vorhersagen, Diagramme und Grenzen. Softwaretests prüfen Berechnungen; zusätzliche Nachprüfungen laden Gewichte erneut und rechnen Ergebniszeilen nach. Die Trennung zwischen mathematisch korrekter Auswertung und medizinischer Aussagekraft bleibt wesentlich.

[Versuchsverzeichnis](EXPERIMENTS.md) · [Budgetvergleich](BUDGET_STUDY.md) · [Augmentationsversuch](AUGMENTATION_STUDY.md)


## Präzisierung und mechanistische Nachanalyse (Versuch 09)

Die Reduktion auf 28×28 Pixel eliminiert radiologische Feinstrukturen. Das synthetische Pixelrauschen dient als kontrolliertes ML-Testbed für niedriges Signal-Rausch-Verhältnis (SNR), nicht als physikalisches Modell klinischer Röntgendegradation. Eine Übertragung auf Dosisänderungen, reale Detektoren oder klinische Bildqualität ist damit nicht belegt. Die fehlenden Patienten-IDs erlauben außerdem keine Absicherung patientenunabhängiger Teilmengen; siehe [Audit und Quellenabgrenzung](AUDIT.md).

[Versuch 09](MECHANISTIC_TIES.md) erklärt alle 250 Modell-Fall-Paare auf den fünf betroffenen Kalibrierungsschwellen durch einen vollständig leeren letzten Merkmalsvektor und den daraus folgenden Bias-Logit. Weder Sättigung bei Konfidenz 1,0 noch eine diskrete Wahrscheinlichkeitskalibrierung verursacht diese Gleichstände. Die 54 leeren Ziehungen entstehen, weil die gezogene Schwelle über sämtliche Konfidenzen der Rauschfälle steigt.

Eine feste Hash-Reihenfolge löst Gleichstände auf: Alle Modelle akzeptieren dann 209 von 262 Kalibrierungsfällen (79,77 %; exakt 80 % ist hier binär nicht möglich). Auf Testdaten garantiert das weder dieselbe Quote noch weniger Fehler. Nullvektoren pauschal zurückzuhalten senkt die Verfügbarkeit teilweise bis auf null und kann die verbleibende Fehlerrate erhöhen. Die Gegenproben begrenzen damit die Interpretation; sie liefern keine neue validierte Sicherheitsregel. Frühere Gewichte und Ergebnisse bleiben erhalten.


## Versuch 10: Transfer ist nicht gleich Kollapsvermeidung

Die Nachanalyse verwendet unveränderte Modelle, Original-Kalibrierung und dieselben Testfälle. Blur und Kontrastreduktion waren nicht Teil des Rauschaugmentationstrainings, aber bereits bekannte Projekt-Testbedingungen. Eine unabhängige OOD-Validierung liegt deshalb nicht vor.

Unter Blur haben augmentierte Modelle mehr Nullvektoren als die Kontrollen, erreichen aber bei allen drei Seeds eine höhere Gesamt-Accuracy. Bei beiden Armen sind die Nullvektorzahlen niedriger als auf Originalbildern; der massive Rausch-Kollaps kehrt unter diesen Störungsstärken nicht wieder. Unter Kontrastreduktion bleiben bei augmentierten Modellen nur 10, 3 und 0 Nullvektoren, während 20,08 %, 22,09 % und 30,89 % ihrer nach Standardregel akzeptierten Vorhersagen falsch sind. Die Kombination aus Tie-Breaking und Nullvektor-Ausschluss beseitigt dieses Problem nicht und kann den verbleibenden Fehleranteil erhöhen.

Nullaktivierung ist somit ein konkreter interner Mechanismus, aber kein allgemeines Maß für Fehlererkennung oder Robustheit. Auch beim bekannten Rauschen hatte Augmentation den Nullvektor nicht vollständig verhindert, sondern den massiven Kollaps deutlich reduziert. Der [Versuch-10-Bericht](TRANSFER_STUDY.md) enthält alle Seeds, beide Annahmeregeln, Originalreferenzen, Abbildungen und die Grenzen der Aussage.
