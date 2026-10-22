"""Answer-only development experiment; every intervention stays at a prompt node."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from tasks import HERE, ROOT, ROLES, SEED, calc, dump, parse, predictions, prompt, sha
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL = ROOT/'followup-readout/checkpoint-replication/model'
RESULTS = HERE/'results'


class Journal:
    def __init__(self, name):
        self.path = RESULTS/f'{name}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row['key'] in self.rows:
                    raise ValueError('Duplicate journal key')
                self.rows[row['key']] = row

    def put(self, key, value):
        if key in self.rows:
            raise ValueError('Already recorded')
        row = dict(key=key, **value)
        with self.path.open('a') as f:
            f.write(json.dumps(row)+'\n')
            f.flush()
        self.rows[key] = row
        return row


def freeze():
    paths = [HERE/'PROTOCOL.md', HERE/'tasks.py', HERE/'run.py',
             HERE/'data/development.jsonl', HERE/'data/reserve.jsonl',
             MODEL/'config.json', MODEL/'tokenizer_config.json']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    path = HERE/'freeze.json'
    if path.exists():
        if json.loads(path.read_text())['files'] != hashes:
            raise RuntimeError('Frozen scientific source changed; new declared version required')
    else:
        dump(path, dict(declared_date='2026-10-01', scope='development, not preregistration', files=hashes))


class Engine:
    def __init__(self, phase):
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        before = torch.cuda.mem_get_info()
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}).eval()
        self.model.requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.genconfig = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=16,
            use_cache=False, eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id,
            bos_token_id=self.tok.bos_token_id, repetition_penalty=1.)
        self.runtime = dict(phase=phase, python=platform.python_version(),
            packages={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'accelerate', 'numpy']},
            model='Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
            model_manifest_sha256=sha(MODEL.parent/'model-manifest.json'),
            cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(),
            cuda_free_total_before=before, cuda_free_total_loaded=torch.cuda.mem_get_info(),
            dtype='bfloat16', attention='eager', use_cache=False, seed=SEED,
            parameters=sum(p.numel() for p in self.model.parameters()),
            generation=self.genconfig.to_dict(), model_config=self.model.config.to_dict(),
            chat_template=self.tok.chat_template, enable_thinking=False)
        dump(RESULTS/f'runtime-{phase}.json', self.runtime)
        print('Loaded local Qwen3-1.7B; no thinking, no cache, greedy answer-only.', flush=True)

    def encode(self, text, spans=None):
        serial = self.tok.apply_chat_template([dict(role='user', content=text)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tok(serial, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids']
        direct = self.tok.apply_chat_template([dict(role='user', content=text)],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        assert ids == direct
        start = serial.index(text)
        positions = {'prompt_end': len(ids)-1}
        for name, end in (spans or {}).items():
            candidates = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b <= start+end and b > start]
            positions[name] = max(candidates)
            assert encoded['offset_mapping'][positions[name]][1] == start+end
        return dict(prompt=text, serialized_prompt=serial, prompt_ids=ids, positions=positions)

    def tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device='cuda')

    @contextlib.contextmanager
    def patch(self, spec=None, stats=None):
        if spec is None:
            yield
            return
        layer, pos, source, rule = spec
        def hook(module, args, output):
            assert isinstance(output, torch.Tensor) and output.ndim == 3
            assert 0 <= pos < output.shape[1]
            if rule == 'read':
                return None
            after = output.clone()
            old = output[0, pos]
            target = old.clone() if rule == 'identity' else source.to(old)
            if rule == 'add':
                target = old + target
            after[0, pos] = target
            if stats is not None and not stats:
                others = after-output
                others[0, pos] = 0
                stats.update(layer=layer, position=pos, delta_norm=float((target-old).float().norm()),
                    write_error=float((after[0, pos]-target).abs().max()),
                    other_positions_max_change=float(others.abs().max()))
            return after
        handle = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def forward(self, ids, keep=1, spec=None, stats=None):
        with self.patch(spec, stats):
            return self.model(input_ids=self.tensor(ids), use_cache=False, logits_to_keep=keep).logits[0].float()

    @torch.inference_mode()
    def capture(self, encoded):
        result = {}
        positions = [encoded['positions'][role] for role in ROLES]
        handles = []
        for layer, module in enumerate(self.model.model.layers):
            def hook(module, args, output, layer=layer):
                result[layer] = output[0, positions].detach().cpu().clone()
            handles.append(module.register_forward_hook(hook))
        try:
            self.forward(encoded['prompt_ids'])
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack([result[i] for i in range(len(handles))])

    @torch.inference_mode()
    def generate(self, encoded, spec=None, forced=None):
        ids = list(encoded['prompt_ids'])
        emitted = list(forced or [])
        stats = {}
        for _ in range(16-len(emitted)):
            scores = self.forward(ids+emitted, spec=spec, stats=stats)[-1]
            token = int(scores.argmax())
            emitted.append(token)
            if token in self.eos:
                break
        text = self.tok.decode(emitted, skip_special_tokens=False)
        return dict(output_ids=emitted, text=text, answer=parse(text),
                    truncated=emitted[-1] not in self.eos, write=stats,
                    forced_output_ids=list(forced or []))

    @torch.inference_mode()
    def score(self, encoded, answer, spec=None):
        target = self.tok.encode(str(answer), add_special_tokens=False) + [self.eos[0]]
        logits = self.forward(encoded['prompt_ids']+target[:-1], keep=len(target), spec=spec)
        lp = logits.log_softmax(-1)
        values = lp[torch.arange(len(target), device='cuda'), torch.tensor(target, device='cuda')].tolist()
        return dict(target=answer, target_ids=target, token_log_probabilities=values,
                    log_probability=sum(values), digits_log_probability=sum(values[:-1]))


def instrument(engine, rows):
    checks = []
    for row in rows[:4]:
        encoded = engine.encode(*prompt(row['recipient'], row['family']))
        ids = encoded['prompt_ids']
        normal = engine.forward(ids)
        vectors = engine.capture(encoded)
        generated = engine.generate(encoded)
        with torch.inference_mode():
            native = engine.model.generate(input_ids=engine.tensor(ids), generation_config=engine.genconfig)[0, len(ids):].tolist()
        c = dict(id=row['id'], native_ids=native, custom_ids=generated['output_ids'],
                 native_equal=native == generated['output_ids'], null_checks=[])
        for layer in [0, 13, 27]:
            for ri, role in enumerate(ROLES):
                for rule in ['read', 'identity', 'replace']:
                    spec = (layer, encoded['positions'][role], vectors[layer, ri], rule)
                    err = float((engine.forward(ids, spec=spec)-normal).abs().max())
                    c['null_checks'].append(dict(layer=layer, role=role, rule=rule, max_logit_abs=err, pass_check=err <= 1e-5))
        c['pass'] = c['native_equal'] and all(x['pass_check'] for x in c['null_checks'])
        checks.append(c)
    encoded = engine.encode(*prompt(rows[0]['recipient']))
    donor = engine.encode(*prompt(rows[0]['donor']))
    source = engine.capture(donor)
    baseline = engine.forward(encoded['prompt_ids'])
    stats = {}
    patched = engine.forward(encoded['prompt_ids'], spec=(13, encoded['positions']['prompt_end'], source[13, 3], 'replace'), stats=stats)
    stats['max_logit_change'] = float((patched-baseline).abs().max())
    positive = stats['delta_norm'] > 0 and stats['write_error'] == 0 and stats['other_positions_max_change'] == 0 and stats['max_logit_change'] > 0
    result = dict(pass_check=all(c['pass'] for c in checks) and positive, cases=checks, positive_write=stats)
    dump(RESULTS/'instrumentation.json', result)
    print('Instrumentation:', result['pass_check'], flush=True)
    return result['pass_check']


def baselines(engine, rows):
    journal = Journal('baselines')
    reuse = {tuple(r['prompt_ids']): r for r in journal.rows.values()}
    for row in rows:
        for condition, side, kind in [('donor_full','donor','full'), ('recipient_full','recipient','full'),
            ('donor_first','donor','first'), ('recipient_first','recipient','first'),
            ('recipient_remaining','recipient','remaining'), ('textual_counterfactual','recipient','counterfactual')]:
            key = row['id']+'/'+condition
            if key in journal.rows:
                continue
            value = row['donor']['z'] if kind == 'counterfactual' else None
            encoded = engine.encode(*prompt(row[side], row['family'], kind, value))
            target = row['predictions']['counterfactual'] if kind == 'counterfactual' else row[side]['z' if kind == 'first' else 'y']
            old = reuse.get(tuple(encoded['prompt_ids']))
            out = engine.generate(encoded) if old is None else {k: old[k] for k in ['output_ids','text','answer','truncated','write','forced_output_ids']}
            saved = journal.put(key, dict(id=row['id'], group=row['group'], condition=condition,
                target=target, correct=out['answer']==target, reused_from=old['key'] if old else None,
                **encoded, **out))
            reuse[tuple(encoded['prompt_ids'])] = saved
        print('Baseline pairs:', len(journal.rows)//6, '/', len(rows), flush=True)
    counts = {}
    for c in ['donor_full','recipient_full','donor_first','recipient_first','recipient_remaining','textual_counterfactual']:
        records = [r for r in journal.rows.values() if r['condition']==c]
        counts[c] = dict(n=len(records), correct=sum(r['correct'] for r in records),
            invalid=sum(r['answer'] is None for r in records), truncated=sum(r['truncated'] for r in records),
            unique_prompts=len({tuple(r['prompt_ids']) for r in records}))
    main = ['donor_full','recipient_full','textual_counterfactual']
    competent = [r['id'] for r in rows if all(journal.rows[r['id']+'/'+c]['correct'] for c in main)]
    gate = all(counts[c]['correct']/counts[c]['n'] >= .9 for c in main) and len(competent)/len(rows) >= .8
    result = dict(pass_check=gate, conditions=counts, triple_competent=competent,
                  n_pairs=len(rows), n_groups=len({r['group'] for r in rows}))
    dump(RESULTS/'competence.json', result)
    print(json.dumps(result, indent=2), flush=True)


def screen(engine, rows):
    journal = Journal('screen')
    for row in rows:
        encoded = engine.encode(*prompt(row['recipient'], row['family']))
        donor = engine.encode(*prompt(row['donor'], row['family']))
        source = engine.capture(donor)
        cf, normal = row['predictions']['counterfactual'], row['recipient']['y']
        unpatched_cf = engine.score(encoded, cf)
        unpatched_r = engine.score(encoded, normal)
        before = unpatched_cf['log_probability']-unpatched_r['log_probability']
        for layer in range(engine.model.config.num_hidden_layers):
            for ri, role in enumerate(ROLES):
                key = f'{row["id"]}/{layer}/{role}'
                if key in journal.rows:
                    continue
                spec = (layer, encoded['positions'][role], source[layer, ri], 'replace')
                cfs = engine.score(encoded, cf, spec)
                rs = engine.score(encoded, normal, spec)
                journal.put(key, dict(id=row['id'], group=row['group'], layer=layer, role=role,
                    donor_position=donor['positions'][role], recipient_position=encoded['positions'][role],
                    counterfactual=cfs, recipient=rs, unpatched_counterfactual=unpatched_cf,
                    unpatched_recipient=unpatched_r,
                    delta_log_ratio=cfs['log_probability']-rs['log_probability']-before))
        print('Screen pairs:', len(journal.rows)//112, '/',len(rows), flush=True)
    sites = []
    for layer in range(engine.model.config.num_hidden_layers):
        for role in ROLES:
            values = [r['delta_log_ratio'] for r in journal.rows.values() if r['layer']==layer and r['role']==role]
            sites.append(dict(layer=layer, role=role, mean_delta_log_ratio=sum(values)/len(values), n=len(values)))
    ranked = sorted(sites, key=lambda s:(-s['mean_delta_log_ratio'],s['layer'],ROLES.index(s['role'])))
    dump(RESULTS/'screen-summary.json', dict(sites=sites, selected=ranked[:3], selection='exploratory, same development pairs'))
    print('Selected exploratory sites:', ranked[:3], flush=True)


def controls(engine, rows):
    selected = json.loads((RESULTS/'screen-summary.json').read_text())['selected']
    journal = Journal('controls')
    baselines = Journal('baselines')
    vector_path = RESULTS/'control-vectors.pt'
    saved_vectors = torch.load(vector_path, weights_only=True) if vector_path.exists() else {}
    conditions = ['donor','unpatched','same_example','isotropic_delta','random_source',
        'same_z_decomposition','same_z_different_answer','different_z_same_answer','text_first_answer_digit']
    group_donor = {row['group']:row['donor'] for row in rows}
    for row in rows:
        encoded = engine.encode(*prompt(row['recipient'], row['family']))
        rv = engine.capture(encoded)
        tasks = dict(donor=row['donor'], random_source=group_donor[(row['group']+1)%len(group_donor)], **row['controls'])
        captures = {c:engine.capture(engine.encode(*prompt(t, row['family']))) for c,t in tasks.items()}
        for site in selected:
            layer, role = site['layer'], site['role']
            ri = ROLES.index(role)
            for condition in conditions:
                key = f'{row["id"]}/{layer}/{role}/{condition}'
                if key in journal.rows:
                    continue
                source_task = tasks.get(condition, row['donor'])
                spec, forced = None, None
                vector = None
                if condition in captures:
                    vector = captures[condition][layer,ri]
                    spec = (layer, encoded['positions'][role], vector, 'replace')
                elif condition == 'same_example':
                    vector = rv[layer,ri]
                    spec = (layer, encoded['positions'][role], vector, 'replace')
                elif condition == 'isotropic_delta':
                    gen = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
                    noise = torch.randn(rv[layer,ri].shape, generator=gen)
                    vector = noise/noise.norm()*(captures['donor'][layer,ri]-rv[layer,ri]).float().norm()
                    spec = (layer, encoded['positions'][role], vector, 'add')
                elif condition == 'text_first_answer_digit':
                    forced = engine.tok.encode(str(row['donor']['y'])[0], add_special_tokens=False)
                out = engine.generate(encoded, spec, forced)
                if condition in ['unpatched','same_example']:
                    assert out['output_ids'] == baselines.rows[row['id']+'/recipient_full']['output_ids'], 'Identity mismatch'
                actual_predictions = predictions(source_task, row['recipient'])
                vector_hash = None
                if vector is not None:
                    saved_vectors[key] = vector
                    vector_hash = hashlib.sha256(vector.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                journal.put(key, dict(id=row['id'], group=row['group'], layer=layer, role=role,
                    condition=condition, recipient_operation=row['recipient']['remaining'],
                    source_task=source_task, recipient_task=row['recipient'], predictions=row['predictions'],
                    actual_source_predictions=actual_predictions, source_hash=vector_hash,
                    counterfactual_success=out['answer']==row['predictions']['counterfactual'],
                    actual_source_success=out['answer']==actual_predictions['counterfactual'],
                    prediction_matches=[k for k,v in row['predictions'].items() if out['answer']==v],
                    **encoded, **out))
        torch.save(saved_vectors, vector_path)
        print('Controlled pairs:',len(journal.rows)//(len(selected)*len(conditions)), '/',len(rows),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['baseline','screen','controls'], required=True)
    args = parser.parse_args()
    freeze()
    rows = [json.loads(line) for line in (HERE/'data/development.jsonl').read_text().splitlines()]
    if args.phase != 'baseline':
        assert json.loads((RESULTS/'instrumentation.json').read_text())['pass_check']
        assert json.loads((RESULTS/'competence.json').read_text())['pass_check']
    engine = Engine(args.phase)
    if args.phase == 'baseline':
        if not instrument(engine, rows):
            raise RuntimeError('Instrumentation gate failed')
        baselines(engine, rows)
    elif args.phase == 'screen':
        screen(engine, rows)
    else:
        controls(engine, rows)
    dump(RESULTS/f'completed-{args.phase}.json', dict(phase=args.phase, completed=True,
        max_cuda_allocated=torch.cuda.max_memory_allocated(), reserve_evaluated=False))


if __name__ == '__main__':
    main()
