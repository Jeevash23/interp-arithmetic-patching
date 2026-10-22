from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
def digest(x):
 return hashlib.sha256(x if isinstance(x,bytes) else json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def freeze(run,config,extra=()):
 p=ROOT/'04-Results'/run;p.mkdir(parents=True,exist_ok=True)
 files=list((ROOT/'02-Code/ai04').glob('*.py'))+list((ROOT/'02-Code').glob('run_*.py'))+[ROOT/'uv.lock']+[ROOT/x for x in extra]
 source={str(f.relative_to(ROOT)):digest(f.read_bytes()) for f in files}
 identity={'config':config,'source':source};h=digest(identity)
 f=p/'freeze.json'
 if f.exists():
  old=readj(f)
  if old['hash']!=h:raise RuntimeError('Refuse changed settings/source in existing run '+str(p))
 else:writej(f,dict(**identity,hash=h,created_utc=utc()))
 return p,h
class Journal:
 def __init__(self,path):
  self.path=Path(path);self.rows={}
  if self.path.exists():
   for l in self.path.read_text().splitlines():
    r=json.loads(l)
    if r['key'] in self.rows:raise ValueError('duplicate journal key')
    self.rows[r['key']]=r
 def add(self,r):
  if r['key'] in self.rows:raise ValueError('overwrite rejected')
  r=dict(r,recorded_utc=utc())
  with self.path.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  self.rows[r['key']]=r
  return r
