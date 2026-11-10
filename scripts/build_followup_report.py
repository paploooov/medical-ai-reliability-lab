"""Build a six-page research report from versioned numerical artifacts.

Requires optional requirements-report.txt; no training and no original images.
"""
from pathlib import Path
import json
import os
import sys
from xml.sax.saxutils import escape
os.environ.setdefault('MPLCONFIGDIR','/tmp/medical-ai-mpl')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))


def build_figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from external_eval import risk_coverage,COVERAGES
    external=np.load(ROOT/'reports/external/predictions.npz')
    mechanistic=np.load(ROOT/'reports/transfer/predictions.npz')
    overview=json.loads((ROOT/'reports/figures/overview_summary.json').read_text())['panel_b']
    p=mechanistic['control_s42_noise_probabilities'];zero=mechanistic['control_s42_noise_zero'];conf=np.maximum(p,1-p)
    assert int(zero.sum())==overview['noise_zero']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    axes[0].hist([conf[zero],conf[~zero]],bins=np.linspace(.5,1,41),stacked=True,
                 color=['#345995','#b9c2cd'],label=['h = 0','h ≠ 0'])
    axes[0].axvline(overview['tau'],color='#bd5735',linestyle='--')
    axes[0].set(title='A. Historical mechanism: control, seed 42',xlabel='Confidence under noise',ylabel='Test cases')
    axes[0].text(.98,.93,f"{zero.sum()}/{len(p)} cases: h = 0\nz = wᵀh + b = b",ha='right',va='top',transform=axes[0].transAxes)
    axes[0].legend(loc='center right')
    for arm,color in [('control_28','#345995'),('augmented_28','#bd5735')]:
        for ds,style in [('internal','--'),('external','-')]:
            y=external[f'labels__{ds}']
            curves=np.array([risk_coverage(y,external[f'{ds}__{arm}__s{s}__clean']) for s in [17,42,83]])
            axes[1].plot(COVERAGES,curves.mean(0),style,color=color,label=f'{arm}, {ds}')
    axes[1].set(title='B. Clean images: internal vs external',xlabel='Coverage',ylabel='Error among retained cases',ylim=(0,.4))
    axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'main_figure.png',dpi=200);fig.savefig(out/'main_figure.pdf');plt.close(fig)


def main():
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    out=ROOT/'reports/followup';out.mkdir(parents=True,exist_ok=True)
    build_figure(out)
    five=json.loads((ROOT/'reports/five_seed/analysis/summary.json').read_text())
    temp=json.loads((ROOT/'reports/temperature_scaling/summary.json').read_text())
    external=json.loads((ROOT/'reports/external/summary.json').read_text())
    views=json.loads((ROOT/'reports/external/view_analysis/summary.json').read_text())
    p=five['primary'];ib=five['fixed_model_image_bootstrap'];ts=temp['results']['control|clean']['nll']
    v=views['contrasts']['control_28|clean|PA_minus_AP']['standardized_retained_error']
    def pct(x):return f'{100*x:.1f} %'
    def pp(x):return f'{100*x:+.1f}'
    pages=[
      ('Konfidenz unter Bildstörungen und Datensatzwechsel',[
       ('p','Medical AI Reliability Lab · Explorative Forschungsnotiz · 10. November 2026 · Kein begutachteter Fachartikel'),
       ('h','Zusammenfassung'),
       ('p','Fragestellung: Wann versagt konfidenzbasierte Ablehnung bei kleinen medizinischen Bildklassifikatoren, und welche Verbesserungen tragen über die untersuchte Störung hinaus? Die Studie verbindet einen mechanistischen Befund auf PneumoniaMNIST, einen kontrollierten Augmentationsvergleich und einen externen Test auf einer fixierten RSNA-Stichprobe.'),
       ('p',f'Zwei zusätzliche, vor dem Training festgelegte Seeds erweitern den internen Vergleich auf fünf Seed-Paare. Der Unterschied des Fehleranstiegs unter Rauschen gegenüber clean, Augmentation minus Kontrolle, beträgt {pp(p["mean"])} Prozentpunkte. Alle fünf Paare zeigen dieselbe Richtung; das approximative 95%-Intervall über Seed-Paare beträgt [{pp(p["ci"][0])}; {pp(p["ci"][1])}] Prozentpunkte.'),
       ('p','Extern steigt die akzeptierte Fehlerrate der ursprünglichen 28px-Kontrolle von 11,2 % auf 20,6 %, bei sinkender Akzeptanz von 78,5 % auf 69,8 %. Rauschaugmentation bietet im vorab definierten Vergleich auf unveränderten externen Bildern keinen Vorteil. Aufnahmeart und Label sind in der Stichprobe stark verknüpft.'),
       ('p',f'Temperature Scaling, ausschließlich intern auf clean-Kalibrierungsbildern angepasst, erhöht die Test-NLL der Kontrolle von {ts["before"]:.3f} auf {ts["after"]:.3f}. Die Konfidenz-Rangfolge bleibt bei positiver Temperatur unverändert. Bessere Fit-Kalibrierung garantiert weder Test-Kalibrierung noch bessere Fehlererkennung.'),
       ('h','Beitrag und Reichweite'),
       ('p','Der Beitrag ist eine nachvollziehbare Verbindung aus konkretem Bias-only-Mechanismus, dokumentierten Grenzen einer Ablehnungsregel und getrennt ausgewiesenen Unsicherheitsquellen. Das Ergebnis ist kein klinischer Sicherheitsnachweis. Niedrige Auflösung, bekannte Testdaten, unklare Patientenunabhängigkeit und offene Rechte des externen Datenspiegels begrenzen die Aussagekraft.'),
       ('p','Aufbau: Methoden (S. 2), Mechanismus und externe Übertragung (S. 3), Kalibrierung und Fallmix (S. 4), fünf Seeds (S. 5), Grenzen, Nachprüfbarkeit und Quellen (S. 6).')]),
      ('Methoden und Trennung der Auswertungen',[
       ('h','Daten und Splits'),
       ('p','PneumoniaMNIST enthält pädiatrische Thoraxaufnahmen. Nach gemeinsamer Pixel-Deduplizierung über die geprüften Auflösungen werden 4.682 Trainingsfälle, 262 Modellwahlfälle, 262 getrennte Kalibrierungsfälle und 621 Testrepräsentanten verwendet. Pixelidentität ist kein Patientenidentifikator. Frühere Auswertungen desselben Benchmarks machen spätere Analysen explorativ.'),
       ('p','Der externe Test umfasst 2.000 RSNA-Bilder: 1.000 Lung Opacity und 1.000 Normal, fixierter Stichprobenseed und eine Aufnahme pro RSNA-ID. Zusätzlich wurden 500 No Lung Opacity / Not Normal getrennt untersucht. Die Stichprobe ist überwiegend erwachsen, enthält jedoch Minderjährige. Lung Opacity ist keine gesicherte Pneumoniediagnose; verschiedene IDs können ohne NIH-Zuordnung nicht als verschiedene Personen garantiert werden.'),
       ('h','Modelle und Intervention'),
       ('p','Das 28px-CNN hat 21.249 Parameter. Kontrolle und Augmentation verwenden dieselbe Architektur, Splits, Adam-Lernrate 0,001, Batchgröße 128 und 45 Epochen. Der niedrigste Modellwahlverlust bestimmt den Checkpoint. Rauschaugmentation betrifft 50 % der Trainingsbilder mit Gaußrauschen SD 0,1; Werte werden auf [0,1] begrenzt. Zu Seeds 17, 42 und 83 kommen 123 und 2026 hinzu. Extern bleiben ausschließlich die ursprünglichen drei Seeds pro Arm und die damaligen Schwellen eingefroren.'),
       ('p','Testbedingungen: clean, Gaußrauschen SD 0,1, Gaußunschärfe Sigma 1 bei 28px, Kontrastreduktion auf die Hälfte um 0,5. Die 64px-Kontrolle des externen Tests ist keine zweite Architektur. ResNet-18 wurde in dieser Arbeit nicht untersucht.'),
       ('h','Endpunkte und Unsicherheit'),
       ('p','Konfidenz ist max(p,1−p). Die Annahmeschwelle ist das linear interpolierte 20%-Quantil der clean-Kalibrierungskonfidenz; Werte auf der Schwelle werden akzeptiert. Akzeptierte Fehlerrate wird zusammen mit Abdeckung berichtet. Risk-Coverage verwendet anteilige Grenzgruppen bei Gleichständen.'),
       ('p','Bild-Bootstrap und Seed-Streuung werden getrennt berichtet: 2.000 gekoppelte Testwiederziehungen bei festen Modellen versus ein approximatives t-Intervall über fünf gepaarte Seed-Kontraste. AP/PA-Intervalle ziehen innerhalb Aufnahmeart×Label. Keines dieser Verfahren erfasst unbekannte Patientenkorrelation oder allgemeine klinische Unsicherheit.')]),
      ('Mechanismus und externer Test',[
       ('img','main_figure.png'),
       ('p','Abbildung 1. Links: historischer Kontrollseed 42 unter Rauschen; 600 von 621 Fällen haben einen Nullvektor in der letzten ReLU-Repräsentation. Rechts: Risk-Coverage auf clean-Bildern, Mittel der drei historischen Seeds; interne Kurven gestrichelt, externe durchgezogen. Dies sind unterschiedliche Analysen: Der linke Einzelfallmechanismus erklärt nicht sämtliche externen Fehler.'),
       ('h','Was mechanistisch nachgewiesen ist'),
       ('p','Für die letzte lineare Schicht gilt z = wᵀh + b. Bei h = 0 folgt exakt z = b, also ein konstanter Output innerhalb dieser Aktivierungsgruppe. Die betroffenen Bias-Werte liefern die Klasse normal, nicht die Mehrheitsklasse. Eine inklusive Quantilschwelle kann eine ganze Punktmasse akzeptieren. Nichtleere Repräsentationen garantieren umgekehrt keine zuverlässige Vorhersage.'),
       ('h','Was der externe Vergleich zeigt'),
       ('p','Die 28px-Kontrolle erreicht extern 73,5 % Accuracy, AUROC 0,820 und ECE 0,141. Ihre akzeptierte Fehlerrate beträgt 20,6 % [18,7; 22,5] bei 69,8 % Akzeptanz. Die Augmentation erreicht 21,8 % akzeptierte Fehler bei 75,2 % Akzeptanz. Der vorab festgelegte primäre Unterschied ist +1,2 Prozentpunkte [+0,4; +1,9], Augmentation minus Kontrolle.'),
       ('p','Unter externem synthetischem Rauschen beträgt die akzeptierte Fehlerrate 45,6 % ohne und 22,5 % mit Augmentation. Ein Nutzen gegen dieselbe Störung überträgt sich damit, ein Vorteil auf unveränderten externen Bildern nicht. Alter, Fallmix, Aufnahmebedingungen und Labeldefinition ändern sich gleichzeitig; die Verschlechterung darf nicht allein dem Alterswechsel zugeschrieben werden.')]),
      ('Wahrscheinlichkeitskalibrierung und Fallmix',[
       ('h','Temperature Scaling'),
       ('p','Pro eingefrorenem 28px-Modell wird eine positive Temperatur innerhalb [0,05; 20] durch Minimierung der binären NLL auf 262 clean-Kalibrierungsfällen angepasst. Testbilder beeinflussen den Fit nicht. Alle sechs Temperaturen liegen zwischen 0,579 und 0,986 und schärfen die Wahrscheinlichkeiten. Keiner der Fits liegt an einer Grenze.'),
       ('p',f'Kontrolle, clean: NLL {ts["before"]:.3f} → {ts["after"]:.3f}; Differenz {ts["difference"]:+.3f}, gepaarter Test-Bootstrap [{ts["ci"][0]:+.3f}; {ts["ci"][1]:+.3f}]. Die NLL steigt in allen acht Arm/Bedingung-Kombinationen. ECE auf clean sinkt leicht, aber ihre Differenzintervalle umfassen null. Die Intervalle konditionieren auf die angepassten Temperaturen und umfassen keine Fit-Unsicherheit.'),
       ('p','Eine nachfolgende Sensitivitätsanalyse zieht Kalibrierungs- und Testfälle gemeinsam neu und passt alle Temperaturen erneut an. Der zentrale 95%-Bereich der clean-NLL-Differenz beträgt bei Kontrolle [−0,022; +0,346] und bei Augmentation [+0,009; +0,765]. Die Einbeziehung des Fits relativiert den konditionalen Befund insbesondere bei der Kontrolle. Dies sind empirische Bereiche unter festen Modellen, keine klinischen Garantien.'),
       ('p','Positive Temperatur erhält die Ordnung nach absolutem Logit und die binäre Entscheidung. Bei entsprechend transformierter Schwelle bleibt dieselbe Menge akzeptiert. Ein Kalibrierungsverfahren kann daher die Wahrscheinlichkeitsskala verändern, ohne die Risk-Coverage-Kurve zu verbessern. Für eine erneute externe NLL-Auswertung fehlen exakte externe Logits; gesättigte gespeicherte Float32-Wahrscheinlichkeiten werden nicht als exakte Logits ausgegeben.'),
       ('h','AP/PA und Labelverteilung'),
       ('p','AP umfasst 964 Fälle (766 positiv, 198 negativ), PA 1.036 Fälle (234 positiv, 802 negativ). Die Positivanteile betragen 79,5 % beziehungsweise 22,6 %. Die akzeptierte Fehlerrate der 28px-Kontrolle ist zunächst 11,9 % bei AP und 30,1 % bei PA.'),
       ('p',f'Bei gleicher Gewichtung beider Klassen innerhalb jeder Ansicht beträgt der Risikoabstand PA minus AP nur noch {pp(v["point"])} Prozentpunkte [{pp(v["ci"][0])}; {pp(v["ci"][1])}]. Das Intervall umfasst null, beweist aber keine Gleichwertigkeit. Standardisierte akzeptierte Fehler werden als Verhältnis gewichteter Fehler und Akzeptanz berechnet; sie sind keine klinische Risikoabschätzung.'),
       ('p','Eine nach Sichtung gewählte Regel AP → positiv, PA → negativ erreicht 78,4 % Accuracy ohne Bildpixel. Das ist eine deskriptive Fallmix-Baseline, kein unabhängig validiertes Modell und kein Beweis dafür, dass die CNNs einen Aufnahmeart-Shortcut verwenden.')]),
      ('Replikation über fünf gepaarte Seeds',[
       ('p',f'Der primäre interne Kontrast beträgt {pp(p["mean"])} Prozentpunkte; alle fünf Seed-Paare sind negativ. Historische drei Seeds: {pp(five["historical_mean"])}; zwei zusätzliche Seeds: {pp(five["additional_mean"])} Prozentpunkte. Es wurden keine ungünstigen Läufe entfernt.'),
       ('p',f'95%-t-Intervall über fünf Seed-Paare: [{pp(p["ci"][0])}; {pp(p["ci"][1])}] Prozentpunkte. Bei fünf Beobachtungen ist dies eine grobe, normalitätsabhängige Beschreibung der Trainingsvariation auf diesem festen Testsatz. Getrennt davon liefert der Bild-Bootstrap bei zehn festen Modellen [{pp(ib["ci"][0])}; {pp(ib["ci"][1])}] Prozentpunkte. Diese Intervalle dürfen nicht als gemeinsame Trainings- und Patientenunsicherheit interpretiert werden.'),
       ('img','../five_seed/analysis/risk_coverage.png'),
       ('p','Abbildung 2. Risk-Coverage für alle vier internen Bedingungen. Linien: Mittel über fünf Seeds; Schattierung: beobachtetes Minimum–Maximum, keine Konfidenzbänder. Die Erweiterung prüft die Robustheit gegenüber weiteren Trainings-Seeds, nicht neue Patienten oder eine neue Architektur.')]),
      ('Grenzen, Nachprüfbarkeit und Werkzeuge',[
       ('h','Was offen bleibt'),
       ('p','Die Studie ist explorativ und nutzt einen wiederholt untersuchten Benchmark mit niedriger Auflösung. Synthetische Störungen repräsentieren keine vollständige reale Aufnahmepipeline. Die fünf Seeds sind endlich und teilen Daten, Architektur und Auswahlregeln. Eine zweite Architektur, neue unabhängige Kliniken und prospektive Annahmeregeln fehlen. Eine Verbesserung unter Gaußrauschen ist keine allgemeine Robustheitsgarantie.'),
       ('p','Die externen Bilder wurden früher über einen inoffiziellen Hugging-Face-Spiegel bezogen. Ein belastbarer Original-Lizenztext und ein Nachweis seiner Weitergabeberechtigung sind bislang nicht verifiziert. Für diese Nachanalysen wurden keine weiteren RSNA-Rohbilder geladen. Prüfsummen sichern Integrität, nicht Nutzungsrechte. Einzelheiten stehen in docs/LICENSING.md.'),
       ('h','Nachprüfbarkeit'),
       ('p','Konfigurationen für Temperature Scaling, stratifizierte Standardisierung und zusätzliche Seeds wurden vor den jeweiligen neuen Berechnungen versioniert. Modelle und numerische Ausgaben früherer Studien bleiben erhalten. Neue Gewichte, Vorhersagen, Logits, Bootstrap-Ausgaben, Quellen-Prüfsummen und Softwareversionen sind gespeichert. scripts/verify_followup.py prüft 120 Kennzahlen separat und die sechs Temperatur-Optima; die Unit-Tests prüfen zusätzlich Randfälle.'),
       ('p','Reproduktion: Anforderungen aus requirements.txt und CPU-PyTorch 2.5.1; Datenprüfung mit scripts/fetch_data.py. Neue Analysen: experiments/temperature_scaling.py, experiments/external_view_analysis.py, experiments/five_seed_analysis.py. Die zusätzlichen Trainingskonfigurationen stehen unter configs/five_seed_*.json. Der Bericht wird mit scripts/build_followup_report.py und optional requirements-report.txt erzeugt.'),
       ('h','Eigenleistung und KI-Assistenz'),
       ('p','Die neuen Auswertungsskripte, Tests, Dokumentation und dieser Bericht wurden in dieser Sitzung mit Codex-Unterstützung erstellt und ausgeführt. Der Auftrag zur Fortführung und zur Forschungsausrichtung kam vom Projektverantwortlichen. Welche früheren methodischen Entscheidungen und Prüfungen er persönlich vorgenommen hat, lässt sich aus dem Repository nicht verlässlich ableiten und wird hier nicht erfunden. Eine fachliche Prüfung durch den Projektverantwortlichen oder eine externe Betreuung wird nicht als erfolgt behauptet.'),
       ('h','Quellen'),
       ('p','Yang et al. (2023), MedMNIST v2. DOI: 10.1038/s41597-022-01721-8. Kermany et al. (2018), Cell. DOI: 10.1016/j.cell.2018.02.010. Shih et al. (2019), Radiology: Artificial Intelligence. DOI: 10.1148/ryai.2019180041. Wang et al. (2017), ChestX-ray8, CVPR. Guo et al. (2017), On Calibration of Modern Neural Networks, ICML, PMLR 70:1321–1330. Numerische Quellen: reports/external/, reports/temperature_scaling/, reports/five_seed/analysis/ und reports/figures/overview_summary.json.')])]
    high_path=ROOT/'reports/high_resolution/summary.json'
    if high_path.exists():
        high=json.loads(high_path.read_text())
        hi=high['results']['native_224'];lo=high['results']['downsample28_upsample224']
        delta=high['native_minus_low_information']['auroc']
        pages[0][1].insert(5,('p',f"Ergänzender Auflösungsvergleich: Ein eingefrorenes ImageNet-ResNet-18 mit trainierter linearer Klassifikationsschicht erhält dieselben Fälle mit 224×224-Information oder auf 28×28 reduzierter Information. Die Test-AUROC beträgt {hi['auroc']:.3f} gegenüber {lo['auroc']:.3f}; gepaarte Differenz {delta['point']:+.3f} [{delta['ci'][0]:+.3f}; {delta['ci'][1]:+.3f}]. Dies prüft Bildinformation unter einem gemeinsamen Encoder, keine klinische Vollauflösung."))
        for i,(kind,text) in enumerate(pages[0][1]):
            pages[0][1][i]=(kind,text.replace('fünf Seeds (S. 5)','Auflösung und fünf Seeds (S. 5)'))
        for i,(kind,text) in enumerate(pages[1][1]):
            pages[1][1][i]=(kind,text.replace('ResNet-18 wurde in dieser Arbeit nicht untersucht.','Ein ergänzender eingefrorener ResNet-18-Encoder wird auf S. 5 untersucht; kein medizinisches End-to-End-Training.'))
        pages[4]=('224px-Information und Trainingsstreuung',[
            ('p','Die 28px-Analysen allein belegen keine diagnostische Aussagekraft. Deshalb wird ein eigener kontrollierter Vergleich auf dem offiziellen 224px-Archiv durchgeführt. Dieselben 4.682/262/262/621 Fälle und ein gemeinsamer eingefrorener ImageNet-ResNet-18-Encoder werden verwendet. Ein Zweig erhält 224px-Information, der andere dieselben Bilder nach Bicubic 224→28→224. Die Eingabegröße ist damit identisch; nur die erhaltene Bildinformation unterscheidet sich.'),
            ('p','Eine standardisierte logistische Klassifikationsschicht wird ausschließlich auf Trainingsmerkmalen gelernt. C ∈ {0,01; 0,1; 1; 10} wird anhand der getrennten Modellwahl-NLL gewählt. Die Konfidenzschwelle folgt dem bisherigen Kalibrierungsquantil. Labelreihenfolge, Prüfsummen und exakte 224px-Duplikate über die behaltenen Splits wurden geprüft. Primärer Endpunkt, vor der Auswertung festgelegt: gepaarte AUROC-Differenz.'),
            ('p',f"224px versus 28px-Information: AUROC {hi['auroc']:.3f} vs. {lo['auroc']:.3f}; Differenz {delta['point']:+.3f} [{delta['ci'][0]:+.3f}; {delta['ci'][1]:+.3f}]. Accuracy {pct(hi['accuracy'])} vs. {pct(lo['accuracy'])}; akzeptierte Fehler {pct(hi['retained_error'])} vs. {pct(lo['retained_error'])} bei Abdeckung {pct(hi['coverage'])} vs. {pct(lo['coverage'])}. Intervalle: 2.000 gepaarte Testwiederziehungen bei festem Encoder und ausgewählten Köpfen."),
            ('img','../high_resolution/risk_coverage.png'),
            ('p',f"Abbildung 2. Auflösungsvergleich, jeweils ein deterministischer linearer Kopf. 224px ist weiterhin eine verkleinerte Darstellung, keine Original-Röntgenauflösung. Separat bestätigt der historische kleine CNN-Vergleich mit fünf Seed-Paaren den Rauschaugmentationseffekt: {pp(p['mean'])} pp; Seed-t-Intervall [{pp(p['ci'][0])}; {pp(p['ci'][1])}], Bild-Bootstrap [{pp(ib['ci'][0])}; {pp(ib['ci'][1])}]. Diese unterschiedlichen Vergleiche dürfen nicht vermischt werden.")])
        for i,(kind,text) in enumerate(pages[5][1]):
            text=text.replace('Eine zweite Architektur, neue unabhängige Kliniken und prospektive Annahmeregeln fehlen.','Der zweite Encoder ist auf ImageNet vortrainiert und bleibt eingefroren; End-to-End-Training bei höherer Auflösung, neue unabhängige Kliniken und prospektive Annahmeregeln fehlen.')
            text=text.replace('experiments/five_seed_analysis.py.','experiments/five_seed_analysis.py und experiments/high_resolution_probe.py.')
            text=text.replace('reports/figures/overview_summary.json.','reports/figures/overview_summary.json; Auflösungsvergleich: reports/high_resolution/.')
            pages[5][1][i]=(kind,text)
    robustness_path=ROOT/'reports/high_resolution/robustness/summary.json'
    if robustness_path.exists():
        robustness=json.loads(robustness_path.read_text())
        primary=robustness['primary'];rn=robustness['results']['native_224|noise_input']
        for i,(kind,text) in enumerate(pages[0][1]):
            if text.startswith('Ergänzender Auflösungsvergleich:'):
                pages[0][1][i]=(kind,text+' Unter Eingangsrauschen steigt die akzeptierte Fehlerrate des 224px-Modells jedoch auf 35,6 % bei 96,8 % Akzeptanz; der clean-Vorteil bedeutet keine allgemein höhere Rauschresistenz.')
        items=pages[4][1]
        items.insert(3,('p',f"Folgetest ohne Refit: gleicher Gaußrausch-Input (SD 0,1), Quellrauschen vor Verkleinerung, Unschärfe und Kontrast. Primärer Kontrast des Rauschen-minus-clean-Risikoanstiegs bei 80 % Abdeckung, 224 minus 28: {primary['point']*100:+.2f} pp [{primary['ci'][0]*100:+.2f}; {primary['ci'][1]*100:+.2f}]. Das Intervall umfasst null. Bei den ursprünglichen Schwellen akzeptiert 224px unter Eingangsrauschen {rn['coverage']['point']:.1%} bei {rn['retained_error']['point']:.1%} Fehlern. Fast alle Fälle werden positiv klassifiziert, mit hoher Konfidenz."))
        for i,(kind,text) in enumerate(items):
            if kind=='img':items[i]=('img','../high_resolution/robustness/summary_figure.png')
            if text.startswith('Abbildung 2.'):
                items[i]=('p',f"Abbildung 2. Links clean, rechts die Kontrolle der Rauschposition. Der gestrichelte 28px-Verlauf enthält Rauschfilterung durch Verkleinerung (RMSE 0,0095 statt 0,0993); er belegt keine allgemeine Überlegenheit von 28px. 224px bleibt unter Originalauflösung. Separat: fünf gepaarte kleine CNNs bestätigen den Augmentationseffekt ({pp(p['mean'])} pp); dies ist eine andere Modellfamilie und kein Robustheitsnachweis für ResNet-18.")
        pages[4]=('Auflösung und Robustheit',items)
        for i,(kind,text) in enumerate(pages[5][1]):
            if text.startswith('Reproduktion:'):
                pages[5][1][i]=(kind,text+' Hochauflösende Robustheit: experiments/high_resolution_robustness.py; scripts/verify_high_resolution_robustness.py prüft alle 2.000 primären Bootstrap-Ziehungen unabhängig nach.')
    augmentation_path=ROOT/'reports/high_resolution/augmentation/summary.json'
    if augmentation_path.exists():
        aug=json.loads(augmentation_path.read_text());effect=aug['primary']
        clean_effect=aug['augmented_minus_control']['clean']['risk_at_80']
        nc=aug['results']['control|noise_42'];na=aug['results']['augmentation|noise_42']
        pages[0][1].insert(6,('p',f"Gezielte Intervention: Ein neuer linearer Kopf mit gleich gewichtetem clean-/Rauschtraining senkt bei unverändertem 224px-Encoder das Risiko unter Testrauschen bei 80 % Abdeckung von {nc['risk_at_80']['point']:.1%} auf {na['risk_at_80']['point']:.1%}. Differenz {effect['point']*100:+.1f} pp [{effect['ci'][0]*100:+.1f}; {effect['ci'][1]*100:+.1f}]. Zwei weitere Testrausch-Seeds bestätigen die Richtung; Unschärfe bleibt ein Gegenbeispiel für allgemeine Zuverlässigkeit."))
        pages[4]=('224px: Verbesserung und Grenzen',[
            ('p','Auflösungsvergleich: Das offizielle 224px-Archiv wird mit derselben auf 28px reduzierten Bildinformation verglichen, jeweils als 224px-Eingabe desselben eingefrorenen ImageNet-ResNet-18. Trainiert wird eine standardisierte logistische Klassifikationsschicht, C-Auswahl nur auf clean-Modellwahlfällen. Alle Fälle und Splits bleiben gleich. AUROC: 0,980 vs. 0,960; gepaarte Differenz +0,019 [0,009; 0,032]. Dies ist keine klinische Vollauflösung.'),
            ('p','Der clean-Vorteil schützt nicht vor Störungen: Das ursprüngliche 224px-Modell akzeptiert unter Eingangsrauschen 96,8 % bei 35,6 % Fehlern. Ein scheinbarer Vorteil des 28px-Zweigs bei Quellrauschen enthält Filterung durch Verkleinerung (RMSE 0,0095 statt 0,0993). Beide Rauschpositionen wurden separat geprüft; Encoder, Köpfe und Schwellen blieben dabei fest.'),
            ('p','Intervention: Jede der 4.682 Trainingsaufnahmen liefert clean- und Rauschmerkmale, jeweils mit Gewicht 0,5; Gesamtgewicht und ursprünglicher StandardScaler bleiben erhalten. Nur ein neuer linearer Kopf wird gelernt. Auswahl aus C ∈ {0,01; 0,1; 1; 10} nach clean-Modellwahl-NLL, Schwelle nur aus clean-Kalibrierung. Trainingsseed 20261116 unterscheidet sich von den Testseeds 42, 7 und 99. Es sind dieselben Testfälle, keine drei unabhängigen Kohorten.'),
            ('p',f"Primär unter Testrauschen Seed 42 bei 80 % Abdeckung: {effect['point']*100:+.2f} pp [{effect['ci'][0]*100:+.2f}; {effect['ci'][1]*100:+.2f}], Augmentation minus Kontrolle. Clean: {clean_effect['point']*100:+.2f} pp [{clean_effect['ci'][0]*100:+.2f}; {clean_effect['ci'][1]*100:+.2f}], kein Nichtunterlegenheitsnachweis. Beide alternativen Rausch-Seeds zeigen ebenfalls geringere Risiken; alle Intervalle sind auf ausgewählte Modelle bedingt."),
            ('img','../high_resolution/augmentation/summary_figure.png'),
            ('p','Abbildung 2. Neue Intervention, beide Modelle mit 224px-Information. Die Verbesserung gilt nicht allgemein: Unter Unschärfe akzeptiert der augmentierte Kopf 99,5 % der Fälle, davon 37,4 % falsch. Bei gleicher 80%-Abdeckung betragen die Blur-Risiken 24,7 % vs. 26,0 %; die unterschiedliche feste Annahmemenge erklärt den scheinbaren Gegensatz. Die Analyse betrifft einen eingefrorenen Encoder und eine Trainingsrausch-Realisierung, kein End-to-End-Training oder externen Transfer.')])
        for i,(kind,text) in enumerate(pages[0][1]):
            pages[0][1][i]=(kind,text.replace('Auflösung und fünf Seeds (S. 5)','Auflösung und Intervention (S. 5)'))
        for i,(kind,text) in enumerate(pages[5][1]):
            if text.startswith('Reproduktion:'):
                text=text.replace('Hochauflösende Robustheit: experiments/high_resolution_robustness.py; scripts/verify_high_resolution_robustness.py prüft alle 2.000 primären Bootstrap-Ziehungen unabhängig nach.','Hochauflösende Folgestudien: experiments/high_resolution_robustness.py und experiments/high_resolution_augmentation.py. Separate Prüfer rechnen die primären Bootstrap-Ziehungen unabhängig nach.')
                pages[5][1][i]=(kind,text)
    regularization_path=ROOT/'reports/high_resolution/augmentation/regularization_sensitivity/summary.json'
    if regularization_path.exists():
        regularization=json.loads(regularization_path.read_text())
        a=regularization['contrasts']['C0.01|noise_42'];b=regularization['contrasts']['C0.1|noise_42']
        pages[4][1].insert(4,('p',f"Post-hoc-Kontrolle der Regularisierung: Der primäre Vergleich verwendete clean-gewählte C-Werte 0,01 und 0,1. Bei gleichem C bleibt der Rauschgewinn bestehen: {a['point']*100:+.1f} pp bei C=0,01 und {b['point']*100:+.1f} pp bei C=0,1; beide bedingten Intervalle liegen unter null. Dies ist eine Sensitivitätsanalyse, kein unabhängiger Bestätigungstest."))
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyResearch',fontName='DejaVu',fontSize=9.5,leading=14,spaceAfter=8,alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='HeadingResearch',fontName='DejaVuBold',fontSize=12,leading=16,spaceBefore=8,spaceAfter=7))
    styles.add(ParagraphStyle(name='TitleResearch',fontName='DejaVuBold',fontSize=19,leading=24,spaceAfter=18))
    story=[];markdown=[]
    for page,(title,items) in enumerate(pages):
        if page:story.append(PageBreak());markdown+=['','---','']
        story.append(Paragraph(escape(title),styles['TitleResearch']));markdown+=['# '+title,'']
        for kind,text in items:
            if kind=='img':
                path=out/text
                from PIL import Image as PILImage
                with PILImage.open(path) as im:w,h=im.size
                width=475;story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,8))
                markdown+=[f'![Abbildung]({text})','']
            else:
                story.append(Paragraph(escape(text),styles['HeadingResearch' if kind=='h' else 'BodyResearch']))
                markdown+=[('## ' if kind=='h' else '')+text,'']
    def footer(canvas,doc):
        canvas.setFont('DejaVu',8);canvas.setFillColor(colors.HexColor('#52616b'))
        canvas.drawString(44,27,'Medical AI Reliability Lab · Explorativ · 10.11.2026')
        canvas.drawRightString(551,27,str(doc.page))
    doc=SimpleDocTemplate(str(out/'research_report.pdf'),pagesize=(595.28,841.89),leftMargin=44,rightMargin=44,
                          topMargin=40,bottomMargin=45,title='Konfidenz unter Bildstörungen und Datensatzwechsel',author='Medical AI Reliability Lab')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'research_report.md').write_text('\n'.join(markdown)+'\n')
    print(out/'research_report.pdf')


if __name__=='__main__':main()
