import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import contextlib,time,re,hashlib,json
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig,LogitsProcessorList,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
from .common import ROOT
from .tasks import parse
DEFAULT=dict(device='cuda',dtype='float32',attention='eager',threads=4,max_tokens=768,temperature=.6,top_p=.95,top_k=20,model_path='../study1-value-transfer-and-readout/models/Qwen3-0.6B')
class Engine:
 def __init__(self,config=None):
  self.c=DEFAULT|dict(config or {})
  torch.set_num_threads(self.c['threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  self.t=AutoTokenizer.from_pretrained(ROOT/self.c['model_path'],local_files_only=True)
  self.m=AutoModelForCausalLM.from_pretrained(ROOT/self.c['model_path'],local_files_only=True,dtype=getattr(torch,self.c['dtype']),attn_implementation=self.c['attention']).to(self.c['device']).eval();self.m.requires_grad_(False)
  self.eos=self.m.generation_config.eos_token_id;self.eos=[self.eos] if isinstance(self.eos,int) else self.eos
  self.proc=LogitsProcessorList([TemperatureLogitsWarper(self.c['temperature']),TopKLogitsWarper(self.c['top_k']),TopPLogitsWarper(self.c['top_p'])])
  self.genconfig=GenerationConfig(do_sample=True,num_beams=1,max_new_tokens=self.c['max_tokens'],use_cache=True,temperature=self.c['temperature'],top_p=self.c['top_p'],top_k=self.c['top_k'],repetition_penalty=1.,eos_token_id=self.eos,pad_token_id=self.t.pad_token_id,bos_token_id=self.t.bos_token_id)
 def tensor(self,ids):return torch.tensor([ids],dtype=torch.long,device=self.c['device'])
 def encode(self,text):return self.t.apply_chat_template([{'role':'user','content':text}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
 def seed(self,ids,seed):return (int(hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:8],16)+seed)%(2**31-1)
 def reset(self,ids,seed,steps=0):
  torch.manual_seed(self.seed(ids,seed));dummy=torch.ones(1,self.m.config.vocab_size,device=self.c['device'])
  for _ in range(steps):torch.multinomial(dummy,num_samples=1)
 @torch.inference_mode()
 def forward(self,ids,cache=None,use_cache=True,keep=1):return self.m(input_ids=self.tensor(ids),past_key_values=cache,use_cache=use_cache,logits_to_keep=keep)
 @contextlib.contextmanager
 def patch(self,layer=None,pos=None,source=None,rule='replace',stats=None):
  if layer is None:yield;return
  def hook(mod,args,out):
   if pos>=out.shape[1]:raise RuntimeError('Absent historical position; no persistent steering allowed')
   if rule=='noop':return None
   before=out[0,pos].detach().clone();after=before.clone() if rule=='identity' else source.to(before) if rule=='replace' else before+source.to(before)
   new=out.clone();new[0,pos]=after
   if stats is not None and not stats:
    diff=(new-out).abs();diff[0,pos]=0
    stats.update(layer=layer,position=pos,before_norm=before.norm().item(),after_norm=after.norm().item(),delta_norm=(after-before).norm().item(),other_positions_max=diff.max().item(),write_error=(new[0,pos]-after).abs().max().item())
   return new
  h=self.m.model.layers[layer].register_forward_hook(hook)
  try:yield
  finally:h.remove()
 @torch.inference_mode()
 def capture(self,ids,layers):
  vectors={};handles=[]
  for layer in layers:
   def h(m,a,o,layer=layer):vectors[layer]=o[0,-1].detach().cpu().clone()
   handles.append(self.m.model.layers[layer].register_forward_hook(h))
  try:self.forward(ids,use_cache=False)
  finally:
   for h in handles:h.remove()
  return vectors
 @torch.inference_mode()
 def generate(self,prompt,seed,prefix=None,layer=None,source=None,rule='replace',pos=None,cache=True,limit=None,trace=False):
  start=time.monotonic();prefix=list(prompt if prefix is None else prefix);pos=len(prefix)-1 if pos is None else pos
  replay=prefix[len(prompt):];budget=self.c['max_tokens']-len(replay)
  if limit is not None:budget=min(budget,limit)
  if budget<=0:raise ValueError('No generation budget')
  self.reset(prompt,seed,len(replay));seq=list(prefix);fresh=[];kv=None;stats={};logits_trace=[]
  for step in range(budget):
   with self.patch(layer if step==0 or not cache else None,pos,source,rule,stats):out=self.forward(seq if step==0 or not cache else fresh[-1:],kv if cache else None,cache)
   logits=out.logits[0,-1].float()
   if trace:logits_trace.append(logits.cpu())
   scores=self.proc(self.tensor(seq),logits[None]);tok=int(torch.multinomial(torch.softmax(scores,dim=-1),num_samples=1)[0,0])
   if cache:kv=out.past_key_values
   fresh.append(tok);seq.append(tok)
   if tok in self.eos:break
  output=replay+fresh;text=self.t.decode(output,skip_special_tokens=False)
  r=dict(prompt_ids=prompt,prefix_ids=prefix,replayed_output_ids=replay,fresh_output_ids=fresh,output_ids=output,text=text,parsed=parse(text),stop_reason='eos' if fresh[-1] in self.eos else 'max_tokens',fresh_tokens=len(fresh),total_output_tokens=len(output),seconds=time.monotonic()-start,seed=seed,effective_seed=self.seed(prompt,seed),write=stats,semantics=dict(layer=layer,position=pos if layer is not None else None,duration='fixed historical node',use_cache=cache,rule=rule if layer is not None else 'none'))
  return (r,logits_trace) if trace else r
 @torch.inference_mode()
 def native(self,prompt,seed):
  self.reset(prompt,seed)
  out=self.m.generate(input_ids=self.tensor(prompt),attention_mask=torch.ones_like(self.tensor(prompt)),generation_config=self.genconfig)
  return out[0,len(prompt):].tolist()
 def cuts(self,r,z,y):
  ids=r['output_ids'];txt=self.t.decode(ids,skip_special_tokens=False);end=txt.rfind('</think>')
  bounds={len(self.t.decode(ids[:i],skip_special_tokens=False)):i for i in range(len(ids)+1)}
  early=re.search(r'(?<!\d)'+str(z)+r'(?!\d)',txt[:end] if end>=0 else txt)
  late=re.search(r'^\s*(?:\*\*)?Intermediate:(?:\*\*)?\s*(-?\d+)',txt[end+8:] if end>=0 else '',re.M)
  final=re.search(r'^\s*(?:\*\*)?Final:(?:\*\*)?\s*(-?\d+)',txt[end+8:] if end>=0 else '',re.M)
  spans={}
  if early:spans['first_z']=early.span()
  if late:spans['label_z']=(end+8+late.start(1),end+8+late.end(1))
  if final:spans['final']=(end+8+final.start(1),end+8+final.end(1))
  cuts={}
  for name,(a,b) in spans.items():
   if a not in bounds or b not in bounds:continue
   i,j=bounds[a],bounds[b]
   if i==0:continue
   cuts[name]=dict(prefix_ids=r['prompt_ids']+ids[:i],output_index=i,numeral_end_index=j,char_start=a,char_end=b,observed_text=txt[a:b],prior_z_mentions=len(re.findall(r'(?<!\d)'+str(z)+r'(?!\d)',txt[:a])),prior_y_mentions=len(re.findall(r'(?<!\d)'+str(y)+r'(?!\d)',txt[:a])),in_thinking=end<0 or a<end,context=txt[max(0,a-160):min(len(txt),b+100)])
  return cuts
 @torch.inference_mode()
 def immediate(self,prefix,zd,zr,layer=None,source=None,rule='replace'):
  with self.patch(layer,len(prefix)-1,source,rule):out=self.forward(prefix,use_cache=False)
  logits=out.logits[0,-1].float();d=self.t.encode(str(zd)[0],add_special_tokens=False);r=self.t.encode(str(zr)[0],add_special_tokens=False)
  assert len(d)==len(r)==1
  return dict(digit_margin=(logits[d[0]]-logits[r[0]]).item(),donor_digit_probability=logits.softmax(-1)[d[0]].item(),recipient_digit_probability=logits.softmax(-1)[r[0]].item(),argmax=int(logits.argmax()),argmax_text=self.t.decode([int(logits.argmax())]))
 @torch.inference_mode()
 def score(self,prefix,values,layer=None,source=None,pos=None,rule='replace',cached=False):
  pos=len(prefix)-1 if pos is None else pos;res={}
  for val in sorted(set(values)):
   cand=self.t.encode(str(val),add_special_tokens=False)+[self.eos[0]]
   if cached:
    with self.patch(layer,pos,source,rule):out=self.forward(prefix)
    lp=[]
    for i,tok in enumerate(cand):
     lp.append(out.logits[0,-1].float().log_softmax(-1)[tok].item())
     if i<len(cand)-1:out=self.forward([tok],out.past_key_values)
   else:
    with self.patch(layer,pos,source,rule):out=self.forward(prefix+cand[:-1],use_cache=False,keep=len(cand))
    logp=out.logits[0].float().log_softmax(-1);lp=[logp[i,t].item() for i,t in enumerate(cand)]
   res[str(val)]=dict(token_ids=cand,token_logprobs=lp,sum_logprob=sum(lp))
  return res
 def random_source(self,before,source,seed):
  gen=torch.Generator(device='cpu').manual_seed(seed);v=torch.randn(before.shape,generator=gen,dtype=before.dtype)
  delta=(source.double()-before.double()).norm()
  return (before.double()+v.double()/v.double().norm()*delta).to(before.dtype)
