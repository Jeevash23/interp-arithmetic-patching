"""Model-specific parts of Study 4: loading a candidate, its chat prompt and its parser.

Everything else is the frozen AI-02 code (code/frozen/, byte-identical copies of Study 3's copies): the write hook,
activation capture, greedy generation without a key-value cache, sequential answer scoring, instrumentation,
behavioural baselines, source baselines and the ten control conditions. The frozen Engine methods are inherited
unchanged; only __init__ (loading) and encode (chat messages) are replaced, and the frozen parser is replaced by the
same strict parser with this model's end strings.
"""
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FROZEN = STUDY / 'code/frozen'
sys.path.insert(0, str(STUDY / 'code'))
sys.path.insert(0, str(FROZEN / 'competence-v2'))
sys.path.insert(0, str(FROZEN))
import study as core  # noqa: E402  frozen competence-v2/study.py (offline mode, kernel cache, package path)
import prompt_formats  # noqa: E402  frozen
from candidates import CANDIDATES, SYSTEM, EXAMPLES, messages, make_parser  # noqa: E402

torch = core.torch
assert SYSTEM == prompt_formats.SYSTEM and [tuple(e) for e in EXAMPLES] == [tuple(e) for e in prompt_formats.EXAMPLES]


def bind_fp8_kernel():
    """The development study's FP8 setup: the pinned matrix kernel and the dense-loader correction."""
    import kernels
    from transformers.integrations import finegrained_fp8
    finegrained_fp8.FP8Experts._impl_tp_layer_overrides.setdefault(None, {})
    kernel = kernels.get_kernel('kernels-community/finegrained-fp8', revision=core.KERNEL_REV)
    finegrained_fp8._FINEGRAINED_FP8 = finegrained_fp8.FineGrainedFP8(
        matmul=kernel.matmul_2d, batched_matmul=kernel.matmul_batched, grouped_matmul=kernel.matmul_grouped)


class Engine(core.Engine):
    def __init__(self, candidate, phase):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, FineGrainedFP8Config
        cfg = CANDIDATES[candidate]
        self.candidate = candidate
        self.format_name = 'fewshot'
        torch.set_num_threads(4); torch.manual_seed(core.SEED); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        before = torch.cuda.mem_get_info()
        extra = {}
        if cfg['precision'] == 'fp8':
            bind_fp8_kernel()
            extra['quantization_config'] = FineGrainedFP8Config()
        folder = STUDY / 'models' / candidate
        self.tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.bfloat16,
                                                          attn_implementation='eager', device_map={'': 'cuda:0'}, **extra).eval()
        self.model.requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.genconfig = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=16, use_cache=False,
                                          eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id,
                                          bos_token_id=self.tok.bos_token_id, repetition_penalty=1.)
        # The frozen generate() calls run.parse; this model's strict parser removes its own end strings.
        core.previous.parse = make_parser(candidate)
        params = list(self.model.parameters())
        runtime = dict(candidate=candidate, phase=phase, repository=cfg['repository'], revision=cfg['revision'],
                       precision=cfg['precision'], quantization=(extra['quantization_config'].to_dict()
                                                                  if 'quantization_config' in extra else None),
                       packages={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'accelerate', 'numpy']},
                       python=platform.python_version(), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                       capability=torch.cuda.get_device_capability(), cuda_free_total_before=before,
                       cuda_free_total_loaded=torch.cuda.mem_get_info(), generation=self.genconfig.to_dict(),
                       model_config=self.model.config.to_dict(), chat_template=self.tok.chat_template,
                       system_prompt=messages(candidate, '')[0]['content'], examples=EXAMPLES,
                       parameters=sum(p.numel() for p in params),
                       dtype_counts={str(d): sum(p.numel() for p in params if p.dtype == d) for d in {p.dtype for p in params}},
                       kernel_revision=core.KERNEL_REV if cfg['precision'] == 'fp8' else None,
                       eos_token_ids=self.eos, end_strings=cfg['end_strings'])
        self.runtime = runtime
        core.dump(core.previous.RESULTS / f'runtime-{phase}.json', runtime)
        print('Loaded', cfg['repository'], 'free GPU', runtime['cuda_free_total_loaded'], flush=True)

    def encode(self, text, spans=None):
        """The frozen FormatEngine.encode with this candidate's messages (system suffix for SmolLM3)."""
        msgs = messages(self.candidate, text)
        serial = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tok(serial, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids']
        assert ids == self.tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=True,
                                                   enable_thinking=False)
        start = serial.rindex(text)
        positions = {'prompt_end': len(ids) - 1}
        for name, end in (spans or {}).items():
            matches = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if start < b <= start + end]
            positions[name] = max(matches)
            assert encoded['offset_mapping'][positions[name]][1] == start + end
        return dict(prompt=text, serialized_prompt=serial, prompt_ids=ids, positions=positions)
