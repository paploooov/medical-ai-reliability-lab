"""Two-page research note from saved preprocessing and transfer results."""
import csv,json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'reports/external_chest224';pre=ROOT/'reports/original_512/preprocessing';s=json.loads((out/'summary.json').read_text());rows=list(csv.DictReader((out/'results.csv').open()))
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));styles=getSampleStyleSheet()
    for style in styles.byName.values():style.fontName='DejaVu'
    styles['BodyText'].fontSize=10;styles['BodyText'].leading=15
    story=[]
    def para(text,style='BodyText'):story.append(Paragraph(text,styles[style]));story.append(Spacer(1,9))
    para('Bildpipeline und externe Übertragung','Title')
    para('Explorative Forschungsnotiz · 11. November 2026 · mit KI-Assistenz erstellt')
    para('1. Unverändertes Modell, andere Vorverarbeitung','Heading2')
    para('Das partiell nachtrainierte 512px-Modell wurde auf denselben 469 Kalibrierungs- und 618 Testfällen unter sechs Bildbedingungen geprüft. Gewichte, Klassifikationsschwelle und Konfidenzschwelle blieben unverändert. Das Protokoll wurde vor der Auswertung festgeschrieben.')
    para('<b>Center Crop ändert 108/618 Entscheidungen: 17,5 % [14,3; 20,7].</b> Das Intervall stammt aus 2.000 gepaarten Wiederziehungen der Dateinamen-Gruppen, bei festem Modell; diese Gruppen sind keine bestätigten Patientenidentitäten.')
    story.append(Image(str(pre/'comparison.png'),width=470,height=205));story.append(Spacer(1,10))
    para('Center Crop erhöht Accuracy von 82,7 % auf 86,6 %, senkt aber Sensitivität von 99,5 % auf 89,4 % und AUROC von 0,9669 auf 0,9217. Bilinear statt Lanczos ändert 24 Entscheidungen und senkt Spezifität von 54,5 % auf 44,6 %. Stretch erreicht 91,1 % Test-Accuracy, fällt auf Kalibrierung jedoch von 98,1 % auf 95,5 %. Keine Pipeline wird anhand dieses bekannten Tests zum Gewinner erklärt.')
    para('Vier synthetische Bild-/Tensorreferenzen decken Rundung, Padding, RGB-Konvertierung und Vergrößerung ab. Python-Pixelidentität und Paketprüfsummen sind geprüft; Apple-Laufzeit und Bildimport bleiben offen. Kein Nachweis von Läsionslokalisierung oder klinischer Eignung.')
    story.append(PageBreak());para('2. Externer Transfer des eingefrorenen 224px-Modells','Heading2')
    n=s['sampling'];para(f"Offizielles ChestMNIST224-Testarchiv, NIH-abgeleitete textbasierte Labels; MD5 geprüft. Nach Pixel-Deduplizierung enthält die feste Stichprobe {n['selected']['pneumonia']} Pneumonie-Label-positive Fälle, {n['selected']['no_finding']} No-finding-Kontrollen und {n['selected']['other_abnormal']} Kontrollen mit anderen Labels. {n['positive_with_additional_tags']} Positive haben zusätzliche Labels.")
    table=[['Kontrollgruppe','AUROC [95 %]','Sens.','Spez.']]
    for row in rows:
        v=s['results'][row['control']]['auroc'];table.append([row['control'],f"{v['point']:.3f} [{v['ci'][0]:.3f}; {v['ci'][1]:.3f}]",f"{float(row['sensitivity'])*100:.1f}%",f"{float(row['specificity'])*100:.1f}%"])
    t=Table(table,colWidths=[135,175,70,70]);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef4')),('BOTTOMPADDING',(0,0),(-1,-1),9)]));story.append(t);story.append(Spacer(1,12))
    para('Primär ist AUROC gegen No-finding-Kontrollen. Sensitivität und Spezifität verwenden die unveränderte Schwelle 0,5. 2.000 labelstratifizierte Bild-Bootstraps je Vergleich erfassen keine Patientenabhängigkeit. Es wurde weder trainiert noch extern kalibriert oder anhand der Testergebnisse ein Modell ausgewählt.')
    r=rows[0];para(f"Die interne Konfidenzschwelle akzeptiert im primären Vergleich {float(r['coverage'])*100:.1f} % der Fälle. Deren Fehlerquote beträgt {float(r['retained_error'])*100:.1f} % gegenüber {100*(1-float(r['accuracy'])):.1f} % über alle Fälle. Auch diese interne Ablehnungsregel überträgt sich hier nicht zuverlässig.")
    para('<b>Geltungsgrenzen:</b> No finding bedeutet nicht bestätigte klinische Gesundheit. Textlabels, Population und Bildpipeline unterscheiden sich gleichzeitig. Die Stichprobe ist fallkontrollartig; Präzision und Kalibrierung sind keine Populationsschätzungen. NIH kann sich mit historischen RSNA-Projekttests überschneiden. Das sind native 224px-Daten und keine unabhängige klinische Validierung des 512px-Modells.')
    para('Quellen: Yang et al., MedMNIST v2, Scientific Data (2023), doi:10.1038/s41597-022-01721-8; Wang et al., ChestX-ray8, CVPR (2017). Offizielles Zenodo-Archiv 10519652; ChestMNIST-Lizenz laut offizieller Metadatei CC BY 4.0. Rohbilder werden hier nicht weiterverteilt.')
    para('Nachrechnung: reports/original_512/preprocessing/ und reports/external_chest224/ enthalten Protokolle, gespeicherte Vorhersagen, Kennzahlen und unabhängige Prüfprotokolle. Offen bleiben mehrere Trainingsseeds, eine neue klinische Originalbild-Kohorte und Xcode-/Gerätetests.')
    SimpleDocTemplate(str(out/'research_note.pdf'),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=42).build(story)
if __name__=='__main__':main()
