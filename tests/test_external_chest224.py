import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from external_chest224 import deduplicate

class ExternalChestTests(unittest.TestCase):
    def test_identical_images_have_one_representative(self):
        x=np.array([np.zeros((3,3)),np.zeros((3,3)),np.ones((3,3))],dtype=np.uint8);labels=np.zeros((3,14),int)
        keep,dups,conflicts=deduplicate(x,labels)
        np.testing.assert_array_equal(keep,[0,2]);self.assertEqual(dups,[[1,0]]);self.assertEqual(conflicts,[])
    def test_conflicting_labels_remove_entire_duplicate_group(self):
        x=np.array([np.zeros((3,3)),np.zeros((3,3)),np.ones((3,3))],dtype=np.uint8);labels=np.zeros((3,14),int);labels[1,6]=1
        keep,_,conflicts=deduplicate(x,labels);np.testing.assert_array_equal(keep,[2]);self.assertEqual(len(conflicts),1)

if __name__=='__main__':unittest.main()
