import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from make_overview import aggregate

class OverviewTests(unittest.TestCase):
    def test_equal_seed_weighting_and_descriptive_range(self):
        rows=[dict(policy='standard',arm='control',condition='noise',seed=s,risk=v) for s,v in zip([17,42,83],[.1,.2,.6])]
        np.testing.assert_allclose(aggregate(rows,'risk','control','noise'),[30,10,60])
    def test_combined_policy_is_not_mixed_into_standard(self):
        rows=[dict(policy='standard',arm='control',condition='noise',seed=s,risk=.2) for s in [17,42,83]]
        rows.append(dict(policy='tie_break_reject_zero',arm='control',condition='noise',seed=17,risk=1))
        self.assertEqual(aggregate(rows,'risk','control','noise'),(20,20,20))
    def test_missing_seed_rejected(self):
        with self.assertRaises(AssertionError):aggregate([],'risk','control','noise')

if __name__=='__main__':unittest.main()
