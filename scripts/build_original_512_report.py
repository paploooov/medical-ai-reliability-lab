"""Three-page data-driven note for the original-image 512px study."""
from pathlib import Path
import json,csv
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports/original_512';s=json.loads((out/'summary.json').read_text());m=json.loads((out/'manifest.json').read_text());large=json.loads((out/'large_encoder/summary.json').read_text());runtime=json.loads((out/'runtime.json').read_text());deployment=json.loads((out/'deployment.json').read_text());rows=list(csv.DictReader((out/'results.csv').open()));rows.append(dict(arm='resnet50_224_probe',**large['metrics']))
    pdfmetrics.registerFont(TTFont('Study','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('StudyBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='B',fontName='Study',fontSize=9,leading=13,spaceAfter=8));styles.add(ParagraphStyle(name='T',fontName='StudyBold',fontSize=18,leading=23,spaceAfter=15));styles.add(ParagraphStyle(name='H',fontName='StudyBold',fontSize=11,leading=15,spaceBefore=10,spaceAfter=7));story=[]
    def p(text,style='B'):story.append(Paragraph(escape(text),styles[style]))
    def img(path,width=495):
        from PIL import Image as PIL
        with PIL.open(path) as im:w,h=im.size
        story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,10))
    def table(data,widths):
        t=Table(data,colWidths=widths);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'StudyBold'),('FONTNAME',(0,1),(-1,-1),'Study'),('FONTSIZE',(0,0),(-1,-1),8),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf4')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,colors.grey)]));story.append(t);story.append(Spacer(1,10))
    names={'mobile_224_probe':'Mobile 224, eingefroren','mobile_512_probe':'Mobile 512, eingefroren','mobile_512_finetuned':'Mobile 512, angepasst','resnet18_224_probe':'ResNet18 224','resnet50_224_probe':'ResNet50 224'}
    p('Originalbilder und 512px: neuer Ausgangsvergleich','T');p('Medical AI Reliability Lab · 11. November 2026 · Explorative Forschungsnotiz')
    p('Fünf Modelle auf derselben bereinigten Originalbild-Kohorte. Zwei Auflösungen, partielles Nachtraining und größere Encoder werden als getrennte Vergleiche untersucht. Modellwahl und Kalibrierung sind vom Test getrennt; keine Variante wird aufgrund des Testergebnisses nachtrainiert.')
    table([['Modell','AUROC','Accuracy','Fehler bei 80 %']]+[[names[r['arm']],f"{float(r['auroc']):.4f}",f"{float(r['accuracy']):.1%}",f"{float(r['risk_at_80']):.1%}"] for r in rows],[225,75,85,110])
    for title,v in [('Primär: Mobile 512 minus 224',s['primary']),('Sekundär: Nachtraining minus eingefroren bei 512',s['contrasts']['mobile_512_finetuned_minus_mobile_512_probe']['auroc']),('Sekundär: ResNet50 minus ResNet18',large['contrast']['auroc'])]:p(f"{title}: AUROC-Differenz {v['point']:+.4f}, 95%-Gruppenbootstrap [{v['ci'][0]:+.4f}; {v['ci'][1]:+.4f}].")
    control=json.loads((out/'head_control/summary.json').read_text());v=control['fine_minus_head']['risk_at_80']
    p(f"Nachtraining senkt den Fehler bei 80 % Abdeckung von 11,8 auf 9,2 %. Gegen eine zusätzliche Kontrolle mit demselben Optimierer, bei der nur der Kopf lernt, beträgt der Unterschied {v['point']*100:+.2f} Prozentpunkte [{v['ci'][0]*100:+.2f}; {v['ci'][1]*100:+.2f}]. Die AUROC verändert sich kaum; bessere Entscheidungsergebnisse sind nicht dasselbe wie bessere Rangordnung.")
    img(out/'model_comparison.png');p('Die Intervalle beruhen auf 2.000 gemeinsamen Wiederziehungen von Dateinamen-Gruppen bei fest ausgewählten Modellen. Sie enthalten weder Trainingsseed-Streuung noch Modellwahlunsicherheit. Sekundäre Vergleiche sind nicht für Mehrfachtests korrigiert.')
    story.append(PageBreak());p('Mehr Bilddetails und sauberere Trennung','T')
    p('Die offiziellen Original-JPEGs stammen aus ChestXRay2017.zip, Kermany, Zhang & Goldbaum (2018), Mendeley Data, DOI 10.17632/rscbjbr9sj.2, CC BY 4.0. Das 1,24-GB-Archiv wurde gegen die veröffentlichte SHA-256-Prüfsumme geprüft. 512px wird aus diesen Originalaufnahmen erzeugt, nicht aus MedMNIST hochskaliert.')
    img(out/'resolution_example.png',width=450)
    table([['Teilmenge','Bilder','Positive','Gruppen']]+[[k,str(v['n']),str(v['positive']),str(v['groups'])] for k,v in m['counts'].items()],[225,90,90,90])
    p('32 exakte Grauwert-Duplikate wurden entfernt. Weitere 453 Trainingsbilder teilen eine aus dem Dateinamen abgeleitete Kennung mit dem Test und wurden ausgeschlossen. Die Kennungen sind keine unabhängig verifizierten Patientenidentitäten; unbekannte Verknüpfungen und ähnliche Bilder können verbleiben.')
    p('Alle 618 Testbilder haben ursprünglich mindestens 728 Pixel an der langen Kante. Die Bilder werden unter Erhalt des Seitenverhältnisses verkleinert und schwarz gepolstert. 55 Nicht-Testbilder besitzen ursprünglich selbst an der langen Kante weniger als 512 Pixel. Auch die höhere Auflösung bleibt meist unter der Originalgröße.')
    p('Dies ist dieselbe Quellfamilie wie MedMNIST. Neue Bereinigung, Aufteilung und Vorverarbeitung verhindern einen direkten Vergleich der absoluten Zahlen mit früheren Studien. Kein neuer externer Standort, keine klinische Einsatzprüfung.')
    story.append(PageBreak());p('Training, Klassenfehler und SwiftUI-Pfad','T')
    selected=s['selection']['mobile_512_finetuned'];p(f"MobileNetV3-Small: {s['models']['mobile_512_probe']['parameters']:,} Parameter mit neuem Kopf. ResNet-50: {large['parameters']:,}. Die eingefrorenen Encoder erhalten neu trainierte lineare Köpfe; C wird nur auf Modellwahl-NLL gewählt. Beim 512px-MobileNet werden zusätzlich die letzten Encoderblöcke angepasst. Gewählte Epoche: {selected['selected_epoch']} von 0–3; Epoche 0 ist die unveränderte Referenz. BatchNorm und der frühe Encoder bleiben fest.")
    table([['Modell','Sensitivität','Spezifität']]+[[names[r['arm']],f"{float(r['sensitivity']):.1%}",f"{float(r['specificity']):.1%}"] for r in rows],[275,110,110])
    p('Die Klassenschwelle ist 0,5. Beim angepassten 512px-Modell fällt die Spezifität von 98,4 % auf Kalibrierungsbildern auf 54,5 % im Test. Eine nachträgliche diagnostische Schwellenanalyse verschärft den Befund: Ein ausschließlich auf Kalibrierungsfällen gewähltes 95%-Spezifitätsziel erreicht im Test nur 52,4 %. Die Ursache dieser Übertragungslücke ist nicht isoliert; diese Schwellen werden nicht in die App übernommen.')
    p('Technischer Weg zur App','H');p(f"Lokaler vollständiger 512px-Checkpoint: {runtime['checkpoint_bytes']/1e6:.2f} MB. Core-ML-Konvertierung: {'erfolgreich' if deployment['converted'] else 'nicht abgeschlossen'}. Das Paket nimmt einen Float32-Grauwerttensor mit 512×512 Pixeln entgegen; Normalisierung liegt im Modell. Eine synthetische Referenzeingabe und ein SwiftUI-Tensorprüfer sind vorbereitet.")
    p('Nicht durchgeführt: Xcode-Build, numerischer Vergleich auf Apple-Hardware, iOS-Bildvorverarbeitung und Geräte-Latenz. Die Linux-CPU-Messungen lassen sich nicht als iPhone-Geschwindigkeit ausgeben. Der Export ist ein Forschungsschritt, keine fertige Diagnose-App.')
    p('Nachprüfbarkeit und Grenzen','H');p('Quelle, Ausschlüsse, Protokolle, Modellwahl, Vorhersagen, Gruppenbootstrap, unabhängige Prüfer und Exportvertrag sind unter reports/original_512/ gespeichert. Code und Analysen wurden mit Codex erstellt und ausgeführt. Eine externe fachliche Prüfung wird nicht behauptet. Nächste Evidenzstufe: unabhängige klinische Kohorten, mehrere Trainingsseeds und Robustheit des angepassten Encoders; größere Auflösung oder Architektur allein garantiert keinen Gewinn.')
    def footer(canvas,doc):
        canvas.setFont('Study',8);canvas.drawString(45,25,'Explorative Forschung · Quellen und Reproduktion: reports/original_512/README.md');canvas.drawRightString(550,25,str(doc.page))
    SimpleDocTemplate(str(out/'research_note.pdf'),pagesize=(595.28,841.89),leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
    print(out/'research_note.pdf')

if __name__=='__main__':main()
