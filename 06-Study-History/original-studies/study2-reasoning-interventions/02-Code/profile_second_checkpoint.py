import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import time,json,torch,resource
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
p=ROOT/'03-Data/raw/models/DeepSeek-R1-Distill-Qwen-1.5B'
t=AutoTokenizer.from_pretrained(p,local_files_only=True);start=time.monotonic();m=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval();m.requires_grad_(False);loadtime=time.monotonic()-start
prompt='Compute z = 31 + 7, then y = z - 4. End with Intermediate: <integer> and Final: <integer>.'
ids=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=False)
torch.manual_seed(948001);start=time.monotonic()
with torch.inference_mode():o=m.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=True,temperature=.6,top_p=.95,top_k=20,max_new_tokens=96)
elapsed=time.monotonic()-start
r=dict(purpose='throughput_only_not_competence',model='DeepSeek-R1-Distill-Qwen-1.5B',device='cpu',dtype='float32',threads=4,load_seconds=loadtime,seconds=elapsed,tokens=o.shape[1]-ids.shape[1],seconds_per_token=elapsed/(o.shape[1]-ids.shape[1]),prompt_ids=ids[0].tolist(),output_ids=o[0,ids.shape[1]:].tolist(),text=t.decode(o[0,ids.shape[1]:]),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,template=t.chat_template,config=m.config.to_dict())
(ROOT/'04-Results/smoke/deepseek-cpu-profile-v1.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
