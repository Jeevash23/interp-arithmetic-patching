from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request
import hashlib,json,datetime,subprocess
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'thought-anchors':'https://arxiv.org/html/2506.19143v4','thought-branches':'https://arxiv.org/html/2510.27484v2','faithfulness-decay':'https://arxiv.org/html/2602.11201v2','sequential-patching':'https://arxiv.org/html/2608.22332v1','legibility':'https://arxiv.org/html/2609.04194v1','scit':'https://arxiv.org/html/2608.27265v1','forking-fast':'https://arxiv.org/html/2608.19611v1','reasoning-to-answer':'https://aclanthology.org/2025.emnlp-main.198.pdf','abstract-math':'https://aclanthology.org/2025.emnlp-main.723.pdf','qwen3-model':'https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/README.md'}
def fetch(item):
 name,url=item
 try:
  data=urlopen(Request(url,headers={'User-Agent':'AI04 research source archive'}),timeout=45).read()
  suffix='.pdf' if url.endswith('.pdf') else '.html' if '/html/' in url else '.md'
  p=ROOT/'01-Papers/sources'/f'{name}{suffix}'
  p.write_bytes(data)
  if suffix=='.pdf':subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
  elif suffix=='.html':
   from html.parser import HTMLParser
   class Extract(HTMLParser):
    def __init__(self):super().__init__();self.out=[];self.skip=0
    def handle_starttag(self,tag,attrs):
     if tag in ['script','style']:self.skip+=1
     if tag in ['p','h1','h2','h3','h4','li','tr']:self.out.append('\n')
    def handle_endtag(self,tag):
     if tag in ['script','style']:self.skip-=1
    def handle_data(self,d):
     if not self.skip:self.out.append(d)
   e=Extract();e.feed(data.decode());p.with_suffix('.txt').write_text(''.join(e.out))
  return dict(name=name,url=url,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except Exception as e:return dict(name=name,url=url,error=repr(e))
if __name__=='__main__':
 records=list(ThreadPoolExecutor(max_workers=4).map(fetch,SOURCES.items()))
 (ROOT/'01-Papers/sources/manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
