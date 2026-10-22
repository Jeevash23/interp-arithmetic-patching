"""Reuse the qualified collector with the explicit CPU checkpoint adapter."""
import runpy,sys
import ai04.engine as engine
from ai04.common import ROOT
from ai04.tasks import parse
SUFFIX=' After your thinking, give exactly two plain-text lines in this format: Intermediate: <integer> followed by Final: <integer>. Do not use boxes, equations, Markdown, or other text after thinking. The first line gives z and the second gives the final answer.'
class PanelEngine(engine.Engine):
 def __init__(self,config=None):super().__init__(dict(model_path='03-Data/raw/models/DeepSeek-R1-Distill-Qwen-1.5B',device='cpu',max_tokens=768)|dict(config or {}))
 def encode(self,text):return super().encode(text+SUFFIX)
 def generate(self,*args,**kwargs):
  out=super().generate(*args,**kwargs);r=out[0] if isinstance(out,tuple) else out;r['parsed']=parse(r['text'].replace('<｜end▁of▁sentence｜>',''));r['parser_normalization']='remove native DeepSeek end-of-sentence marker only';return out
engine.Engine=PanelEngine
sys.argv=[str(ROOT/'02-Code/run_experiment.py'),'05-Notes/protocols/second-panel-v1.json']
runpy.run_path(sys.argv[0],run_name='__main__')
