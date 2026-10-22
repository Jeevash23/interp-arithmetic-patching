"""Four bounded HTTP transfers of one pinned public file, validated by SHA256."""
import concurrent.futures,hashlib,json,math,os,shutil,threading,time
from pathlib import Path
import httpx
from huggingface_hub import get_hf_file_metadata,hf_hub_url
HERE=Path(__file__).resolve().parent
REPO='Qwen/Qwen3-4B-Instruct-2507-FP8'
REV='8591804019c8b22094c3b5b4454e0edc05dffc98'
target=HERE/'models/qwen4b'
partial=target/'model.safetensors.partial'
state=HERE/'download-ranges-state.json'
metadata=get_hf_file_metadata(hf_hub_url(REPO,'model.safetensors',revision=REV),token=False)
total=metadata.size;expected=metadata.etag.strip('"');chunk=64*1024**2
assert len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
if state.exists():
    record=json.loads(state.read_text());assert record['revision']==REV and record['expected_sha256']==expected
else:
    old=sorted((target/'.cache/huggingface/download').glob('*.incomplete'),key=lambda p:p.stat().st_size,reverse=True)
    completed=[]
    if old and old[0].stat().st_size:
        initial=old[0].stat().st_size;assert initial<=total
        old[0].rename(partial);completed=list(range(initial//chunk))
    record=dict(repository=REPO,revision=REV,total_bytes=total,expected_sha256=expected,
                chunk_bytes=chunk,completed=completed,method='4 parallel HTTP range requests')
completed=set(record['completed']);lock=threading.Lock()
allocated=partial.stat().st_blocks*512 if partial.exists() else 0
assert shutil.disk_usage(HERE).free-(total-allocated)>1024**3
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
                with client.stream('GET',metadata.location,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','Cache-Control':'no-cache'}) as response:
                    assert response.status_code==206, f'Expected range response, got {response.status_code}'
                    assert response.headers['content-range']==f'bytes {start}-{end}/{total}',response.headers.get('content-range')
                    offset=start
                    for data in response.iter_bytes(chunk_size=1024**2):
                        assert offset+len(data)<=end+1
                        written=os.pwrite(fd,data,offset);assert written==len(data);offset+=len(data)
                    assert offset==end+1,f'Truncated range {i}'
            with lock:
                completed.add(i);save()
                print('Verified ranges',len(completed),'/',math.ceil(total/chunk),flush=True)
            return
        except (httpx.HTTPError,AssertionError) as error:
            if attempt==3:raise
            print('Retry range',i,type(error).__name__,flush=True);time.sleep(min(2**attempt,8))

try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,[i for i in range(math.ceil(total/chunk)) if i not in completed]))
    os.fsync(fd)
finally:os.close(fd)
with partial.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual==expected,'Provider SHA256 mismatch'
partial.rename(target/'model.safetensors')
files=[]
for p in sorted(target.iterdir()):
    if p.is_file():
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        files.append(dict(name=p.name,bytes=p.stat().st_size,sha256=digest))
(HERE/'model-manifest-qwen4b.json').write_text(json.dumps(dict(repository=REPO,revision=REV,
    files=files,client_date='2026-10-01',weight_matches_provider_sha256=True),indent=2)+'\n')
print('Complete: pinned checkpoint SHA256 verified.',flush=True)
