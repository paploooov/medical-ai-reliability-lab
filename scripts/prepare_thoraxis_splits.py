"""Metadata-only, protocol c44117a: exact grouped random splits."""
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack, csr_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/thoraxis_splits'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def main():
    cfg = json.loads((ROOT/'configs/thoraxis_preparation.json').read_text())
    source = ROOT/'reports/original_512/manifest.json'
    records = json.loads(source.read_text())['records']
    groups = sorted({r['group'] for r in records})
    gi = {g:i for i,g in enumerate(groups)}
    parts = cfg['partition_order']; n = len(groups); nv = n*4
    target = {(p,c):sum(r['split']==p and r['label']==c for r in records) for p in parts for c in (0,1)}
    counts = np.zeros((n,2), dtype=int)
    for r in records: counts[gi[r['group']],r['label']] += 1
    a = lil_matrix((n+8,nv)); rhs = np.ones(n+8)
    for i in range(n): a[i,i*4:i*4+4] = 1
    for j,p in enumerate(parts):
        for c in (0,1):
            a[n+j*2+c,j::4] = counts[:,c]
            rhs[n+j*2+c] = target[p,c]
    a = a.tocsr()
    def solve(cost, matrix=a, low=rhs, high=rhs, bounds=None):
        result = milp(cost, integrality=np.ones(nv), bounds=bounds or Bounds(0,1),
                      constraints=LinearConstraint(matrix,low,high),
                      options={'time_limit':cfg['solver_time_limit_seconds'], 'mip_rel_gap':0.0})
        if result.status != 0: raise RuntimeError(f'STOP: solver status {result.status}: {result.message}')
        return result
    OUT.mkdir(exist_ok=True)
    allsets = {}; summaries = []
    for seed in cfg['split_seeds']:
        start=time.perf_counter()
        cost=np.random.Generator(np.random.PCG64(seed)).random(nv)
        result=solve(cost); x=np.rint(result.x).astype(int)
        # Exclude first assignment to detect a numerically equal alternative optimum.
        alt_a=vstack([a,csr_matrix(x.reshape(1,-1))],format='csr')
        alt=solve(cost,alt_a,np.r_[rhs,-np.inf],np.r_[rhs,n-1])
        gap=float(alt.fun-result.fun)
        if gap <= cfg['objective_tie_tolerance']:
            # Lexicographic smallest partition sequence on the optimal objective face.
            face=vstack([a,csr_matrix(cost.reshape(1,-1))],format='csr')
            lo=np.r_[rhs,-np.inf]; hi=np.r_[rhs,result.fun+cfg['objective_tie_tolerance']]
            lb=np.zeros(nv); ub=np.ones(nv)
            for i in range(n):
                lex=np.zeros(nv); lex[i*4:i*4+4]=np.arange(4)
                r=solve(lex,face,lo,hi,Bounds(lb,ub))
                chosen=int(np.argmax(r.x[i*4:i*4+4])); lb[i*4+chosen]=1
                ub[i*4:i*4+4]=0; ub[i*4+chosen]=1
            x=lb.astype(int)
        assert np.array_equal(a@x,rhs)
        assign={g:parts[int(np.argmax(x[i*4:i*4+4]))] for i,g in enumerate(groups)}
        rows=[dict(r,split=assign[r['group']]) for r in records]
        sets={p:{r['name'] for r in rows if r['split']==p} for p in parts}
        assert len(set(r['name'] for r in rows))==len(rows)==5371
        assert len(set(r['pixel_sha256'] for r in rows))==len(rows)
        for p,q in itertools.combinations(parts,2): assert not sets[p]&sets[q]
        for g in groups: assert len({r['split'] for r in rows if r['group']==g})==1
        for p,c in target: assert sum(r['split']==p and r['label']==c for r in rows)==target[p,c]
        ah=hashlib.sha256(json.dumps(assign,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        summary={'seed':seed,'seconds':time.perf_counter()-start,'assignment_sha256':ah,
                 'test_original_test':sum(r['split']=='test' and r['original_split']=='test' for r in rows),
                 'counts':{p:{str(c):target[p,c] for c in (0,1)} for p in parts},
                 'alternative_objective_gap':gap,'objective':float(cost@x),'checks_passed':True}
        path=OUT/f'{seed}.json'
        save(path,{'protocol_commit':'c44117a','source_sha256':sha(source),
                   'implementation_sha256':sha(Path(__file__)), 'scipy':scipy.__version__,
                   'solver':'SciPy bundled HiGHS; version fixed by scipy wheel',**summary,'records':rows})
        summary['manifest_sha256']=sha(path); summaries.append(summary); allsets[seed]=sets
        print(json.dumps(summary),flush=True)
    pairs=[]
    for s,t in itertools.combinations(allsets,2):
        a,b=allsets[s]['test'],allsets[t]['test']
        pairs.append({'seeds':[s,t],'test_intersection':len(a&b),'test_jaccard':len(a&b)/len(a|b),
                      'intersection_by_partition':{p:len(allsets[s][p]&allsets[t][p]) for p in parts}})
    save(OUT/'checks.json',{'splits':summaries,'pairs':pairs,
        'test_union':len(set.union(*(s['test'] for s in allsets.values()))),
        'test_common_all':len(set.intersection(*(s['test'] for s in allsets.values())))})

if __name__=='__main__': main()
