"""Presentation-only fix: move ROC legend away from historical-comparison text."""
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'reports/original_512/thoraxis_30/stage3_source_correlation.svg'
ET.register_namespace('','http://www.w3.org/2000/svg')
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
tree=ET.parse(path)
legend=next(node for node in tree.getroot().iter() if node.attrib.get('id')=='legend_1')
legend.set('transform','translate(0,-45)')
tree.write(path,encoding='utf-8',xml_declaration=True)
