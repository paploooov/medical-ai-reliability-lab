import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('train',Path(__file__).resolve().parents[1]/'experiments/train.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MetricsTests(unittest.TestCase):
    def test_confident_errors_denominator(self):
        r=m.metrics(np.array([0,0,1,1]),np.array([.1,.95,.8,.2]))
        self.assertEqual(r['accuracy'],.5)
        self.assertEqual(r['high_confidence_error'],.25)
        self.assertEqual(r['error_among_confident'],.5)
        self.assertEqual(r['sensitivity'],.5)
        self.assertEqual(r['specificity'],.5)
    def test_empty_confident_subset(self):
        r=m.metrics(np.array([0,1]),np.array([.4,.6]))
        self.assertIsNone(r['error_among_confident'])
        self.assertEqual(r['high_confidence_count'],0)
if __name__=='__main__': unittest.main()
