"""Left-padded independent fixed-node interventions, requiring serial-equivalence validation."""
import time
from .engine import torch
from .tasks import parse

@torch.inference_mode()
def generate_batch(engine,jobs,trace=False):
    start=time.monotonic();n=len(jobs)
    if not n:return []
    prefixes=[j.get('prefix_ids',j['prompt_ids']) for j in jobs]
    replays=[p[len(j['prompt_ids']):] for p,j in zip(prefixes,jobs)]
    lengths=[len(p) for p in prefixes];width=max(lengths)
    budgets=[engine.config['max_new_tokens']-len(r) for r in replays]
    if min(budgets)<=0:raise ValueError('No generation budget remains')
    padded=[[engine.tokenizer.pad_token_id]*(width-len(p))+p for p in prefixes]
    ids=torch.tensor(padded,device=engine.device,dtype=torch.long)
    mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prefixes],device=engine.device,dtype=torch.long)
    positions=(mask.cumsum(-1)-1).clamp_min(0)
    generators=[]
    dummy=torch.ones(1,engine.model.config.vocab_size,device=engine.device)
    for j,replay in zip(jobs,replays):
        g=torch.Generator(device=engine.device).manual_seed(engine.sampling_seed(j['prompt_ids']))
        if engine.config['do_sample']:
            for _ in replay:torch.multinomial(dummy,num_samples=1,generator=g)
        generators.append(g)
    del dummy
    writes=[{} for _ in jobs];handles=[]
    for layer in sorted({j['layer'] for j in jobs if j.get('layer') is not None}):
        group=[i for i,j in enumerate(jobs) if j.get('layer')==layer]
        def hook(module,args,output,layer=layer,group=group):
            if not isinstance(output,torch.Tensor) or output.ndim!=3:raise TypeError(type(output))
            result=output.clone()
            for i in group:
                j=jobs[i];before=output[i,-1].detach();rule=j.get('rule','replace')
                if rule in ['noop','read']:continue
                if rule=='identity':after=before.clone()
                elif rule=='zero':after=torch.zeros_like(before)
                elif rule=='replace':after=j['source'].to(before)
                elif rule=='add':after=before+j['source'].to(before)
                else:raise ValueError(rule)
                result[i,-1]=after
                writes[i]={'before_norm':float(before.norm()),'after_norm':float(after.norm()),'delta_norm':float((after-before).norm()),'write_error':float((result[i,-1]-after).abs().max()),'other_positions_max_change':0.0,'shape':[1,lengths[i],output.shape[-1]],'padded_batch_shape':list(output.shape),'position':lengths[i]-1,'padded_position':width-1,'layer':layer}
            return result
        handles.append(engine.model.model.layers[layer].register_forward_hook(hook))
    fresh=[[] for _ in jobs];traces=[[] for _ in jobs];alive=[True]*n;cache=None
    try:
        for step in range(max(budgets)):
            output=engine.model(input_ids=ids,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True,logits_to_keep=1)
            if step==0:
                for h in handles:h.remove()
                handles=[]
            cache=output.past_key_values
            next_tokens=[]
            for i,j in enumerate(jobs):
                if not alive[i]:next_tokens.append(engine.tokenizer.pad_token_id);continue
                logits=output.logits[i,-1].float()
                if trace:traces[i].append(logits.detach().cpu())
                if engine.processors is None:token=int(logits.argmax())
                else:
                    sequence=engine.tensor(prefixes[i]+fresh[i])
                    scores=engine.processors(sequence,logits[None,:])
                    token=int(torch.multinomial(torch.softmax(scores,dim=-1),num_samples=1,generator=generators[i])[0,0])
                fresh[i].append(token);next_tokens.append(token)
                if token in engine.eos or len(fresh[i])>=budgets[i]:alive[i]=False
            if not any(alive):break
            ids=torch.tensor(next_tokens,device=engine.device,dtype=torch.long)[:,None]
            mask=torch.cat([mask,torch.tensor(alive,device=engine.device,dtype=torch.long)[:,None]],dim=-1)
            positions=torch.tensor([l+step for l in lengths],device=engine.device,dtype=torch.long)[:,None]
    finally:
        for h in handles:h.remove()
    elapsed=time.monotonic()-start;results=[]
    for i,j in enumerate(jobs):
        ids_out=replays[i]+fresh[i];text=engine.tokenizer.decode(ids_out,skip_special_tokens=False)
        layer=j.get('layer')
        result={'prompt_ids':j['prompt_ids'],'prefix_ids':prefixes[i],'replayed_output_ids':replays[i],'fresh_output_ids':fresh[i],'output_ids':ids_out,'text':text,'parsed':parse(text),'truncated':fresh[i][-1] not in engine.eos,'stop_reason':'eos' if fresh[i][-1] in engine.eos else 'max_new_tokens','fresh_tokens':len(fresh[i]),'total_output_tokens':len(ids_out),'seconds':elapsed/n,'batch_wall_seconds':elapsed,'batch_size':n,'use_cache':True,'write':writes[i],'sampling_seed':engine.sampling_seed(j['prompt_ids']),
                'semantics':{'layer':layer,'hook':None if layer is None else f'model.layers.{layer}.output','recipient_position':lengths[i]-1 if layer is not None else None,'generation_step':len(replays[i]),'rule':j.get('rule','replace') if layer is not None else 'none','duration':'one fixed historical node','cached_write_count':1 if layer is not None else None,'dtype':str(engine.dtype),'padding':'left, masked, original unpadded position IDs'}}
        results.append((result,traces[i]) if trace else result)
    return results
