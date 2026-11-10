import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from high_resolution_probe import low_information,audit_data


class HighResolutionTests(unittest.TestCase):
    def test_low_information_removes_fine_structure_without_changing_shape(self):
        fine=((np.indices((224,224)).sum(axis=0)%2)*255).astype(np.uint8)
        reduced=low_information(fine)
        self.assertEqual(reduced.shape,fine.shape)
        self.assertEqual(reduced.dtype,np.uint8)
        self.assertLess(reduced.std(),fine.std()/10)
        self.assertLess(abs(reduced.mean()-fine.mean()),1)

    def test_pixel_audit_rejects_cross_split_duplicates_and_label_mismatch(self):
        data={};small={}
        for i,split in enumerate(['train','val','test']):
            data[split+'_images']=np.full((1,224,224),i,dtype=np.uint8)
            data[split+'_labels']=np.array([[i%2]])
            small[split+'_labels']=data[split+'_labels'].copy()
        audit=audit_data(data,small,{'train':[0],'selection':[0],'test':[0]})
        self.assertEqual(audit['exact_duplicates_in_retained_cases'],0)
        data['test_images']=data['train_images'].copy()
        with self.assertRaises(ValueError):audit_data(data,small,{'train':[0],'test':[0]})
        small['val_labels']=np.array([[0]])
        with self.assertRaises(AssertionError):audit_data(data,small,{'train':[0]})

if __name__=='__main__':unittest.main()
