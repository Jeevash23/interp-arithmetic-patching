"""Download the prespecified fallback at an immutable revision with bounded I/O."""
import concurrent.futures,hashlib,json,math,os,shutil,threading,time
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_OFFLINE']='0'
import httpx
from huggingface_hub import HfApi,hf_hub_download,hf_hub_url,get_hf_file_metadata
HERE=Path(__file__).resolve().parent
REPO='Qwen/Qwen2.5-1.5B-Instruct';REV='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
target=HERE/'models/qwen25';target.mkdir(parents=True,exist_ok=True)
meta=HfApi(token=False).model_info(REPO,revision=REV,files_metadata=True)
files=[f for f in meta.siblings if '/' not in f.rfilename and (f.rfilename.endswith(('.json','.safetensors','.txt','.model')) or f.rfilename in ['README.md','LICENSE'])]
remaining=sum(max(0,(f.size or 0)-(target/(f.rfilename+'.partial')).stat().st_blocks*512) if (target/(f.rfilename+'.partial')).exists() else (f.size or 0) for f in files if not (target/f.rfilename).exists())
assert shutil.disk_usage(HERE).free-remaining>1024**3,'Disk headroom below 1 GiB'
print('Pinned fallback',REPO,REV,'remaining bytes',remaining,flush=True)
checks={}
for item in files:
    name=item.rfilename
    if not name.endswith('.safetensors'):
        hf_hub_download(REPO,name,revision=REV,local_dir=target,token=False)
        continue
    md=get_hf_file_metadata(hf_hub_url(REPO,name,revision=REV),token=False)
    expected=md.etag.strip('"');total=md.size;chunk=64*1024**2
    assert len(expected)==64 and total==item.size
    path=target/name;partial=target/(name+'.partial');state=HERE/('download-qwen25-'+name+'-state.json')
    if not path.exists():
        record=json.loads(state.read_text()) if state.exists() else dict(repository=REPO,revision=REV,file=name,total_bytes=total,expected_sha256=expected,chunk_bytes=chunk,completed=[])
        assert record['revision']==REV and record['expected_sha256']==expected
        completed=set(record['completed']);lock=threading.Lock()
        fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,total)
        def save():
            record['completed']=sorted(completed)
            tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(state)
        save()
        def fetch(i):
            start=i*chunk;end=min((i+1)*chunk,total)-1
            for attempt in range(4):
                try:
                    with httpx.Client(follow_redirects=True,timeout=httpx.Timeout(90.,connect=30.)) as client:
                        with client.stream('GET',md.location,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','Cache-Control':'no-cache'}) as response:
                            assert response.status_code==206
                            assert response.headers['content-range']==f'bytes {start}-{end}/{total}'
                            offset=start
                            for block in response.iter_bytes(chunk_size=1024**2):
                                assert offset+len(block)<=end+1
                                written=os.pwrite(fd,block,offset);assert written==len(block);offset+=len(block)
                            assert offset==end+1
                    with lock:
                        completed.add(i);save();print('Verified ranges',len(completed),'/',math.ceil(total/chunk),flush=True)
                    return
                except (httpx.HTTPError,AssertionError) as error:
                    if attempt==3:raise
                    print('Retry',i,type(error).__name__,flush=True);time.sleep(min(2**attempt,8))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(fetch,[i for i in range(math.ceil(total/chunk)) if i not in completed]))
            os.fsync(fd)
        finally:os.close(fd)
        with partial.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==expected,'Provider weight hash mismatch'
        partial.rename(path)
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    assert actual==expected
    checks[name]=True
manifest=[]
for p in sorted(target.iterdir()):
    if p.is_file():
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        manifest.append(dict(name=p.name,bytes=p.stat().st_size,sha256=digest))
(HERE/'model-manifest-qwen25.json').write_text(json.dumps(dict(repository=REPO,revision=REV,files=manifest,client_date='2026-10-01',weights_match_provider_sha256=checks),indent=2)+'\n')
print('Complete: fallback weights match provider SHA256.',flush=True)
