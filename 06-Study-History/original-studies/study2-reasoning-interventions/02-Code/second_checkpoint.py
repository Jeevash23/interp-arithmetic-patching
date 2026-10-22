from ai04.engine import Engine
from ai04.tasks import parse
class SecondEngine(Engine):
 def __init__(self):super().__init__(dict(model_path='03-Data/raw/models/DeepSeek-R1-Distill-Qwen-1.5B',device='cpu',max_tokens=768))
 def generate(self,*args,**kwargs):
  out=super().generate(*args,**kwargs);r=out[0] if isinstance(out,tuple) else out
  r['parsed']=parse(r['text'].replace('<｜end▁of▁sentence｜>',''))
  r['parser_normalization']='remove native DeepSeek end-of-sentence marker only'
  return out
