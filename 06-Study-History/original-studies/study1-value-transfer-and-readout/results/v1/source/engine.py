import contextlib
import hashlib
import json
import os
import re
import time

from .tasks import ROOT, parse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class Engine:
    def __init__(self, config=None):
        self.config=config or json.loads((ROOT/'configs/pilot.json').read_text())
        torch.set_num_threads(self.config['threads'])
        torch.manual_seed(self.config['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        self.device=self.config['device']
        self.dtype=getattr(torch,self.config['dtype'])
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-0.6B', local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen3-0.6B', local_files_only=True, dtype=self.dtype, attn_implementation=self.config['attention']).to(self.device).eval()
        self.model.requires_grad_(False)
        self.eos=self.model.generation_config.eos_token_id
        self.eos=[self.eos] if isinstance(self.eos,int) else self.eos
        self.generation_config=GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=self.config['max_new_tokens'], use_cache=True, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=self.eos, pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id)
        record={"config":self.config,"native_generation_config":self.model.generation_config.to_dict(),"explicit_native_comparison_config":self.generation_config.to_dict(),"model_config":self.model.config.to_dict(),"chat_template":self.tokenizer.chat_template,"parameters":sum(p.numel() for p in self.model.parameters()),"eos_ids":self.eos,"eos_tokens":self.tokenizer.convert_ids_to_tokens(self.eos),"initial_cuda_free_total":torch.cuda.mem_get_info()}
        (ROOT/'results/model-runtime.json').write_text(json.dumps(record,indent=2))
        print(f'Loaded native Qwen3: {record["parameters"]} parameters, {self.dtype}, {self.config["attention"]}',flush=True)

    def encode(self,text):
        return self.tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=self.config['enable_thinking'])

    def tensor(self,ids):
        return torch.tensor([ids],dtype=torch.long,device=self.device)

    @torch.inference_mode()
    def forward(self,ids,cache=None,use_cache=True):
        return self.model(input_ids=self.tensor(ids),past_key_values=cache,use_cache=use_cache,logits_to_keep=1)

    @contextlib.contextmanager
    def intervention(self,layer,position,source=None,rule="replace",stats=None):
        def hook(module,args,output):
            if not isinstance(output,torch.Tensor) or output.ndim!=3:
                raise TypeError(f'Unexpected block output {type(output)}')
            if position>=output.shape[1]:
                raise RuntimeError('Historical position absent: do not apply this hook to incremental decoding')
            before=output[0,position].detach().clone()
            if rule=="noop":
                return None
            if rule=="read":
                if stats is not None: stats['read_norm']=float(before.norm())
                return None
            result=output.clone()
            if rule=="identity":
                after=before.clone()
            elif rule=="zero":
                after=torch.zeros_like(before)
            elif rule=="add":
                after=before+source.to(before)
            elif rule=="replace":
                after=source.to(before)
            else:
                raise ValueError(rule)
            result[0,position]=after
            if stats is not None and not stats:
                stats.update(before_norm=float(before.norm()),after_norm=float(after.norm()),delta_norm=float((after-before).norm()),write_error=float((result[0,position]-after).abs().max()),shape=list(output.shape),position=position,layer=layer)
                left=(result[:,:position]-output[:,:position]).abs().max().item() if position else 0.0
                right=(result[:,position+1:]-output[:,position+1:]).abs().max().item() if position+1<output.shape[1] else 0.0
                stats['other_positions_max_change']=max(left,right)
            return result
        handle=self.model.model.layers[layer].register_forward_hook(hook)
        try: yield
        finally: handle.remove()

    @torch.inference_mode()
    def capture(self,ids,layers,position=None):
        position=len(ids)-1 if position is None else position
        vectors={}; handles=[]
        for layer in layers:
            def read(module,args,output,layer=layer):
                assert isinstance(output,torch.Tensor) and output.ndim==3
                vectors[layer]=output[0,position].detach().cpu().clone()
            handles.append(self.model.model.layers[layer].register_forward_hook(read))
        try: self.forward(ids,use_cache=False)
        finally:
            for handle in handles: handle.remove()
        return vectors

    def cuts(self,record):
        prompt_ids=record['prompt_ids']; output=record['output_ids']
        text=self.tokenizer.decode(output,skip_special_tokens=False)
        cuts={"prompt_end":prompt_ids}
        m=re.search(r'^Intermediate:\s*(-?\d+)\s*$',text,re.M)
        f=re.search(r'^Final:\s*(-?\d+)\s*$',text.replace('<|im_end|>',''),re.M)
        # Use original tokens, never tokenize a reconstructed/gold continuation.
        boundaries=[len(self.tokenizer.decode(output[:i],skip_special_tokens=False)) for i in range(len(output)+1)]
        if m:
            start,end=m.span(1)
            before=max(i for i,n in enumerate(boundaries) if n<=start)
            after=next((i for i,n in enumerate(boundaries) if n>=end),None)
            if before>0: cuts['pre_z']=prompt_ids+output[:before]
            if after is not None: cuts['post_z']=prompt_ids+output[:after]
        if f:
            before=max(i for i,n in enumerate(boundaries) if n<=f.start(1))
            if before>0: cuts['pre_final']=prompt_ids+output[:before]
        return cuts

    @torch.inference_mode()
    def generate(self,prompt_ids,prefix_ids=None,layer=None,source=None,rule="replace",use_cache=True,trace=False,position=None):
        start=time.monotonic()
        prefix_ids=prompt_ids if prefix_ids is None else prefix_ids
        prompt_len=len(prompt_ids)
        replay=prefix_ids[prompt_len:]
        budget=self.config['max_new_tokens']-len(replay)
        if budget<=0: raise ValueError('No fresh-generation budget remains')
        position=len(prefix_ids)-1 if position is None else position
        stats={}; new=[]; traces=[]; cache=None
        seq=list(prefix_ids)
        for step in range(budget):
            ids=seq if step==0 or not use_cache else [new[-1]]
            intervene=layer is not None and (step==0 or not use_cache)
            ctx=self.intervention(layer,position,source,rule,stats) if intervene else contextlib.nullcontext()
            with ctx:
                out=self.forward(ids,cache=cache if use_cache else None,use_cache=use_cache)
            logits=out.logits[0,-1]
            if trace: traces.append(logits.detach().cpu())
            token=int(logits.argmax())
            if use_cache: cache=out.past_key_values
            new.append(token); seq.append(token)
            if token in self.eos: break
        output_ids=list(replay)+new
        text=self.tokenizer.decode(output_ids,skip_special_tokens=False)
        result={"prompt_ids":prompt_ids,"prefix_ids":prefix_ids,"replayed_output_ids":replay,"fresh_output_ids":new,"output_ids":output_ids,"text":text,"parsed":parse(text),"truncated":new[-1] not in self.eos,"stop_reason":"eos" if new[-1] in self.eos else "max_new_tokens","fresh_tokens":len(new),"total_output_tokens":len(output_ids),"seconds":time.monotonic()-start,"use_cache":use_cache,"write":stats,
                "semantics":{"layer":layer,"hook":None if layer is None else f'model.layers.{layer}.output',"recipient_position":position if layer is not None else None,"generation_step":len(replay),"rule":rule if layer is not None else "none","duration":"one fixed historical node","cached_write_count":1 if layer is not None and use_cache else None,"dtype":str(self.dtype)}}
        if trace: return result,traces
        return result

    @torch.inference_mode()
    def native_generate(self,ids):
        output=self.model.generate(input_ids=self.tensor(ids),attention_mask=torch.ones_like(self.tensor(ids)),generation_config=self.generation_config)
        return output[0,len(ids):].tolist()

def vector_hash(vector):
    return hashlib.sha256(vector.float().numpy().tobytes()).hexdigest()
