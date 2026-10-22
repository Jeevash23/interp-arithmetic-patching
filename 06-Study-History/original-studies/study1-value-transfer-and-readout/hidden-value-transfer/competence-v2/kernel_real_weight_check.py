"""Post-baseline numerical diagnostic, without changing scientific model settings."""
import os,sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.packages'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_CACHE']=str(HERE/'.cache/huggingface')
os.environ['TRITON_CACHE_DIR']=str(HERE/'.cache/triton');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,kernels
from safetensors import safe_open
torch.manual_seed(20261003);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
kernel=kernels.get_kernel('kernels-community/finegrained-fp8',revision='8c178950e8710e26a2210e4e909cd02dc16c8715')
records=[]
with torch.inference_mode(),safe_open(HERE/'models/qwen4b/model.safetensors',framework='pt',device='cpu') as f:
    for name in ['model.layers.0.self_attn.q_proj','model.layers.18.mlp.up_proj','model.layers.35.mlp.down_proj']:
        w=f.get_tensor(name+'.weight').cuda();s=f.get_tensor(name+'.weight_scale_inv').cuda()
        n,k=w.shape;x=torch.randn(76,k,device='cuda',dtype=torch.bfloat16)
        # Independently apply the declared per-row, per-128-feature FP8 quantization.
        xg=x.float().reshape(76,k//128,128);sx=xg.abs().amax(-1,keepdim=True)/448
        xq=((xg/sx.clamp_min(1e-12)).to(torch.float8_e4m3fn).float()*sx).reshape(76,k)
        wq=w.float()*s.repeat_interleave(128,0).repeat_interleave(128,1)[:n,:k]
        ref=(xq@wq.T).to(torch.bfloat16)
        y=kernel.matmul_2d(x,w,s,[128,128],torch.bfloat16)
        yshort=kernel.matmul_2d(x[:73].contiguous(),w,s,[128,128],torch.bfloat16)
        rec=dict(layer_module=name,weight_shape=list(w.shape),scale_min=float(s.min()),scale_max=float(s.max()),
            rms_relative_to_fp32_quantized_reference=float((y.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt()),
            max_absolute_to_reference=float((y-ref).abs().max()),
            prefix_shape_change_max_absolute=float((y[:73]-yshort).abs().max()),
            reference='FP32 matmul with separately quantized/dequantized input and weights, cast to BF16',
            finite=bool(torch.isfinite(y).all()))
        records.append(rec);print(json.dumps(rec),flush=True)
        del w,s,x,xg,sx,xq,wq,ref,y,yshort
(HERE/'kernel-real-weight-check.json').write_text(json.dumps(dict(stage='post-baseline numerical diagnostic',seed=20261003,records=records),indent=2)+'\n')
