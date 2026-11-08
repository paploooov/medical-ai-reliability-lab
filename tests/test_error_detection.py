import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from error_detection import detection, risk_curve
import unittest
import numpy as np
class DetectionTests(unittest.TestCase):
    def test_errors_rank_above_correct(self):
        r=detection(np.array([0,1,0,1]),np.array([.01,.99,.6,.4]),.8)
        self.assertEqual(r['error_detection_auroc'],1)
        self.assertEqual(r['error_recall'],1)
        self.assertEqual(r['retained_error_rate'],0)
    def test_confident_mistakes_reverse_ranking(self):
        r=detection(np.array([0,1,0,1]),np.array([.99,.01,.4,.6]),.8)
        self.assertEqual(r['error_detection_auroc'],0)
    def test_ties_are_retained_together(self):
        c,r=risk_curve(np.array([0,1]),np.array([.9,.9]))
        self.assertEqual(c,[1]);self.assertEqual(r,[.5])
    def test_no_errors(self):
        r=detection(np.array([0,1]),np.array([.1,.9]),.8)
        self.assertIsNone(r['error_detection_auroc'])
        self.assertIsNone(r['error_recall'])
if __name__=='__main__':unittest.main()
