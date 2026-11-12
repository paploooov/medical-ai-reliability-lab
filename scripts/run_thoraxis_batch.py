"""Resumable serial background batch. Saves every attempt; commits only stage outputs."""
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/thoraxis_training_batch.json'
LOCAL=ROOT/'data/thoraxis_source_shift/training_batch'
STATE=LOCAL/'state.json'
SUMMARY=ROOT/'reports/original_512/thoraxis_training_30.csv'
LOGS=ROOT/'reports/original_512/thoraxis_training_30_logs.json'
REPORT=ROOT/'docs/THORAXIS_TRAINING.md'
OWNED=[str(p.relative_to(ROOT)) for p in (SUMMARY,LOGS,REPORT)]

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def verify_plan(plan):
    for path,expected in plan['input_sha256'].items():
        if sha(ROOT/path)!=expected: raise RuntimeError(f'Input changed: {path}')
        committed=subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest()!=expected: raise RuntimeError(f'Input not committed: {path}')
    if subprocess.check_output(['git','show','HEAD:configs/thoraxis_training_batch.json'],cwd=ROOT)!=PLAN.read_bytes():
        raise RuntimeError('Batch plan not committed')

def emit(plan,state):
    rows=[]
    for run in plan['runs']:
        item=state['runs'][run['run_id']]; result=item.get('result',{})
        rows.append(dict(run_id=run['run_id'],split=run['split'],training_seed=run['seed'],status=item['status'],
                         attempts=len(item['attempts']),selected_epoch=result.get('selected_epoch',''),
                         selection_nll=result.get('selected_selection_nll',''),runtime_seconds=result.get('total_seconds',''),
                         checkpoint_sha256=result.get('checkpoint_sha256',''),checkpoint_path=item.get('checkpoint_path','')))
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);SUMMARY.write_text(f.getvalue())
    save(LOGS,{'protocol':'c44117a','status':state['status'],'updated_at':now(),'runs':state['runs'],
               'batch_error':state.get('batch_error'), 'medical_test_inference':False})
    done=[r for r in rows if r['status']=='completed']; failures=sum(a.get('status') in ('failed','interrupted') for v in state['runs'].values() for a in v['attempts'])
    seconds=sum(float(r['runtime_seconds']) for r in done)
    REPORT.write_text(f'''# THORAXIS: 30 Trainingsläufe ohne Testinferenz

Protokoll: `c44117a`. Status: **{state['status']}**, **{len(done)}/30** Modelle abgeschlossen. Stand (UTC): {now()}.

Fünf offizielle Referenzmodelle und fünf gemischte Aufteilungen mit jeweils denselben Seeds 17, 42, 83, 123 und 2026. Alle Modelle starten neu vom ursprünglichen ImageNet-Encoder. Der frühere technische Probelauf bleibt getrennt. Training, Modellwahl und Kalibrierung verwenden nur die eigenen Entwicklungsdaten. Der medizinische Kopf erhält keine Testzeilen.

Die [vollständige Tabelle aller 30 Läufe](../reports/original_512/thoraxis_training_30.csv) enthält gewählte Epoche, Modellwahl-NLL, Laufzeit, Checkpoint-SHA-256 und lokalen Gewichtspfad. [Logs aller Versuche](../reports/original_512/thoraxis_training_30_logs.json) enthalten Epochenverlauf und Abbruchgründe. Bisher **{failures}** fehlgeschlagene/unterbrochene Versuche; Summe der erfolgreichen Laufzeiten: **{seconds/3600:.2f} Stunden**. Abgebrochene Versuche werden erhalten und mit demselben Seed wiederholt, höchstens dreimal insgesamt; danach wird der offene Lauf markiert.

Gewichte liegen nach bestehender Regel im ignorierten Verzeichnis `data/thoraxis_source_shift/training_batch/`; Logs und Hashes werden nach jedem Lauf committet. Keine Gewichte werden hochgeladen. Der Auftrag läuft seriell im Hintergrund, speichert seinen Fortschritt und beendet sich nach dem Paket. Wiederaufnahme: mit derselben gesperrten Python-Umgebung `python scripts/run_thoraxis_batch.py`; vorhandene erfolgreiche Checkpoints werden anhand ihres Hashes geprüft und nicht neu trainiert.

## Grenzen

**Keine Testinferenz und keine neue Aussage über Testleistung oder Kausalität.** Die Modellwahl-NLL beschreibt ausschließlich die Selection-Menge. Lokale Gewichte sind noch kein öffentlich abrufbares Release. Bei Eingabe-/Hashfehlern stoppt der Auftrag; ein Prozessabbruch erfordert Wiederaufnahme mit dem obigen Befehl. Details eines möglichen Blockers stehen in den Logs.
''')
    save(STATE,state)

def commit_outputs(state):
    # --only keeps unrelated staged user changes out of this commit.
    subprocess.run(['git','add','--',*OWNED],cwd=ROOT,check=True)
    diff=subprocess.run(['git','diff','--quiet','HEAD','--',*OWNED],cwd=ROOT)
    if diff.returncode:
        subprocess.run(['git','commit','--only','-m','Save THORAXIS batch training progress (no test inference)','--',*OWNED],cwd=ROOT,check=True)
    save(LOCAL/'last_commit.json',{'commit':git('rev-parse','HEAD'),'status':state['status'],'at':now()})

def initialize(plan):
    LOCAL.mkdir(parents=True,exist_ok=True)
    if STATE.exists(): return json.loads(STATE.read_text())
    state={'status':'prepared','created_at':now(),'runs':{r['run_id']:{'status':'pending','attempts':[]} for r in plan['runs']}}
    emit(plan,state);return state

def main():
    plan=json.loads(PLAN.read_text());LOCAL.mkdir(parents=True,exist_ok=True)
    with (LOCAL/'batch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=initialize(plan)
        if '--initialize' in sys.argv: return
        save(LOCAL/'process.json',{'pid':os.getpid(),'started_at':now(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        try:
            verify_plan(plan)
            state['status']='running';state.pop('batch_error',None);emit(plan,state)
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            for run in plan['runs']:
                item=state['runs'][run['run_id']]
                if item['status']=='completed':
                    if sha(ROOT/item['checkpoint_path'])!=item['result']['checkpoint_sha256']: raise RuntimeError('Saved checkpoint hash mismatch')
                    continue
                for old in item['attempts']:
                    if old['status']=='running':
                        old.update(status='interrupted',reason='Supervisor ended before saving completion',ended_at=now())
                        log=ROOT/old['stdout_path']
                        old['stdout']=log.read_text(errors='replace') if log.exists() else ''
                while len(item['attempts'])<plan['max_attempts_per_run']:
                    verify_plan(plan)
                    attempt_dir=LOCAL/run['run_id']/f"attempt_{len(item['attempts'])+1}"
                    attempt_dir.mkdir(parents=True,exist_ok=False)
                    output=attempt_dir/'artifacts';log=attempt_dir/'stdout.log'
                    attempt={'number':len(item['attempts'])+1,'status':'running','started_at':now(),'stdout_path':str(log.relative_to(ROOT))}
                    item['attempts'].append(attempt);item['status']='running';emit(plan,state)
                    command=[sys.executable,str(ROOT/'scripts/train_thoraxis_protocol.py'),'--manifest',str(ROOT/run['manifest']),
                             '--seed',str(run['seed']),'--run-id',run['run_id'],'--output',str(output)]
                    started=time.monotonic();reason=None
                    try:
                        with log.open('w') as stream:
                            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=plan['timeout_seconds'])
                        if result.returncode: raise RuntimeError(f'Exit code {result.returncode}; see stdout')
                        measured=json.loads((output/'training_run.json').read_text());checkpoint=output/'checkpoint.pt'
                        if measured['seed']!=run['seed'] or measured['run_id']!=run['run_id']: raise RuntimeError('Run identity mismatch')
                        if measured['test_evaluated'] or measured['medical_test_predictions']!=0: raise RuntimeError('Test inference flag violated')
                        if measured['checkpoint_sha256']!=sha(checkpoint): raise RuntimeError('Checkpoint corrupted')
                        if measured['manifest_sha256']!=plan['input_sha256'][run['manifest']]: raise RuntimeError('Split hash changed')
                        item.update(status='completed',result=measured,checkpoint_path=str(checkpoint.relative_to(ROOT)))
                    except Exception:
                        reason=traceback.format_exc();item['status']='retry_pending'
                    attempt.update(status='failed' if reason else 'completed',ended_at=now(),wall_seconds=time.monotonic()-started,
                                   reason=reason,stdout=log.read_text(errors='replace'),stdout_sha256=sha(log))
                    emit(plan,state);commit_outputs(state)
                    if not reason: break
                if item['status']!='completed':
                    item['status']='failed';emit(plan,state);commit_outputs(state)
            state['status']='completed' if all(r['status']=='completed' for r in state['runs'].values()) else 'incomplete'
            emit(plan,state);commit_outputs(state)
        except BaseException:
            state['status']='blocked';state['batch_error']=traceback.format_exc();emit(plan,state)
            try:commit_outputs(state)
            except Exception:traceback.print_exc()
            raise

if __name__=='__main__':main()
