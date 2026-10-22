import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json,time,sys,hashlib,platform,importlib.metadata
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
modelpath=ROOT.parent/'study1-value-transfer-and-readout/models/Qwen3-0.6B'
t=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
m.requires_grad_(False)
records=[]
for i,prompt in enumerate(['Compute this two-stage program. First, z = 37 + 8. Then, y = z - 6. Show your calculations. End with exactly these two lines: Intermediate: <integer> and Final: <integer>.','The intermediate has been supplied: z = 62. Use that value directly. Compute y = z - 6. End with Final: <integer>.']):
 ids=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=True,return_tensors='pt',return_dict=False).to('cuda')
 torch.manual_seed(940000+i);start=time.monotonic()
 with torch.inference_mode():out=m.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=768,do_sample=True,temperature=.6,top_p=.95,top_k=20,repetition_penalty=1.,use_cache=True)
 elapsed=time.monotonic()-start
 r=dict(prompt=prompt,prompt_ids=ids[0].tolist(),output_ids=out[0,ids.shape[1]:].tolist(),text=t.decode(out[0,ids.shape[1]:],skip_special_tokens=False),seconds=elapsed,tokens=out.shape[1]-ids.shape[1]);records.append(r);print(json.dumps(r),flush=True)
record=dict(records=records,cuda=torch.cuda.get_device_name(),peak_memory_bytes=torch.cuda.max_memory_allocated(),python=sys.version,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','scipy','pandas','matplotlib','safetensors','accelerate','tokenizers']},chat_template=t.chat_template,model_config=m.config.to_dict(),generation_config=m.generation_config.to_dict(),parameters=sum(p.numel() for p in m.parameters()))
(ROOT/'04-Results/smoke/native-profile-v1.json').write_text(json.dumps(record,indent=2))
print('Completed smoke',record['peak_memory_bytes'])
