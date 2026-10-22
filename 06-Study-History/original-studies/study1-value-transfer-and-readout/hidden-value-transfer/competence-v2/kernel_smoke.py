"""Check the pinned official Triton FP8 implementation on the actual GPU."""
import os,sys,json,inspect,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.packages'))
os.environ['HF_HUB_CACHE']=str(HERE/'.cache/huggingface')
os.environ['TRITON_CACHE_DIR']=str(HERE/'.cache/triton')
os.environ['HF_HUB_OFFLINE']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,kernels
torch.manual_seed(20261001)
torch.use_deterministic_algorithms(True)
module=kernels.get_kernel('kernels-community/finegrained-fp8',revision='8c178950e8710e26a2210e4e909cd02dc16c8715')
record=dict(gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
    kernels=kernels.__version__,module_file=module.__file__,
    revision='8c178950e8710e26a2210e4e909cd02dc16c8715',
    functions={n:str(inspect.signature(getattr(module,n))) for n in ['matmul_2d','matmul_batched','matmul_grouped']})
x=(torch.randn(4,2560,device='cuda')*.1).to(torch.bfloat16)
w=(torch.randn(2560,2560,device='cuda')*2).to(torch.float8_e4m3fn)
scales=torch.full((20,20),.01,device='cuda',dtype=torch.float32)
y=module.matmul_2d(x,w,scales,[128,128],torch.bfloat16)
y2=module.matmul_2d(x,w,scales,[128,128],torch.bfloat16)
reference=torch.nn.functional.linear(x,(w.float()*.01).to(torch.bfloat16)).float()
relative_rms=float((y.float()-reference).square().mean().sqrt()/reference.square().mean().sqrt())
record.update(finite=bool(torch.isfinite(y).all()),repeat_max_abs=float((y-y2).abs().max()),
    relative_rms_vs_dequantized_weight_bf16=relative_rms,
    pass_check=bool(torch.isfinite(y).all()) and torch.equal(y,y2) and relative_rms<.1)
print(json.dumps(record,indent=2),flush=True)
(HERE/'kernel-inspection.json').write_text(json.dumps(record,indent=2)+'\n')
