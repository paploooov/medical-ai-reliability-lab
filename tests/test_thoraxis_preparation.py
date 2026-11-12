"""Independent checks of saved preparation artifacts; no image/model access."""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class SplitArtifacts(unittest.TestCase):
    def test_exact_partition_and_provenance(self):
        source=ROOT/'reports/original_512/manifest.json'
        original=json.loads(source.read_text())['records']
        by_id={r['name']:r for r in original}
        target=collections.Counter((r['split'],r['label']) for r in original)
        checks=json.loads((ROOT/'configs/thoraxis_splits/checks.json').read_text())
        sets={}
        for summary in checks['splits']:
            path=ROOT/f"configs/thoraxis_splits/{summary['seed']}.json"
            m=json.loads(path.read_text()); rows=m['records']
            self.assertEqual(sha(path),summary['manifest_sha256'])
            self.assertEqual(sha(source),m['source_sha256'])
            self.assertEqual(len(rows),len(by_id))
            self.assertEqual({r['name'] for r in rows},set(by_id))
            self.assertEqual(collections.Counter((r['split'],r['label']) for r in rows),target)
            groups=collections.defaultdict(set); pixels=collections.defaultdict(set)
            for r in rows:
                self.assertEqual({k:v for k,v in r.items() if k!='split'},
                                 {k:v for k,v in by_id[r['name']].items() if k!='split'})
                groups[r['group']].add(r['split']);pixels[r['pixel_sha256']].add(r['split'])
            self.assertTrue(all(len(v)==1 for v in groups.values()))
            self.assertTrue(all(len(v)==1 for v in pixels.values()))
            assignment={g:next(iter(v)) for g,v in groups.items()}
            ah=hashlib.sha256(json.dumps(assignment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.assertEqual(ah,summary['assignment_sha256'])
            sets[summary['seed']]={r['name'] for r in rows if r['split']=='test'}
            self.assertEqual(summary['test_original_test'],sum(by_id[n]['original_split']=='test' for n in sets[summary['seed']]))
        self.assertEqual(sorted(sets),list(range(20261112,20261117)))
        for pair in checks['pairs']:
            a,b=(sets[s] for s in pair['seeds'])
            self.assertEqual(len(a&b),pair['test_intersection'])
            self.assertEqual(len(a&b)/len(a|b),pair['test_jaccard'])
        self.assertEqual(len(checks['pairs']),10)
        self.assertEqual(len(set.union(*sets.values())),checks['test_union'])
        self.assertEqual(len(set.intersection(*sets.values())),checks['test_common_all'])

if __name__=='__main__':unittest.main()
