"""Study 5 interventions (PROTOCOL.md sections 5 to 7), run inside Modal on one GPU per worker.

The model, prompts, generation and write hook are the frozen AI-02 code (code/frozen/, byte-identical copies of Study
3's copies): greedy decoding without a key-value cache, the write reapplied at its prompt position at every step, the
few-shot box-wording prompt. New here: a hook that writes several positions at one block (span writes), capture at
arbitrary positions, the class means of the fitting problems, and the work lists of the stages. Every generation is
one record in a gzipped JSON-lines file per pair; a file is written once, after its pair is complete.
"""
import contextlib
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FROZEN = STUDY / 'code/frozen'
sys.path.insert(0, str(FROZEN / 'competence-v2'))
sys.path.insert(0, str(FROZEN))
import study as core  # noqa: E402  frozen competence-v2/study.py (offline mode, kernel cache, package path, Engine)
from tasks import ROLES, calc, prompt  # noqa: E402  frozen

torch = core.torch
LEAD = 'A box starts with '
SPANS = ['post_operand', 'first_step']
SITES3 = [(23, 'prompt_end'), (20, 'a_end'), (9, 'a_end'), (10, 'a_end'), (14, 'b_end'), (20, 'first_stage_end')]
SPAN_SOURCES = ['same_z_decomposition', 'same_z_different_answer', 'different_z_same_answer',
                'same_leading_different_value', 'random_source']


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def seeded_noise(key, shape, norm):
    """The frozen isotropic_delta recipe: a seeded standard normal draw scaled to the given norm."""
    gen = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    noise = torch.randn(shape, generator=gen)
    return noise / noise.norm() * norm


class Engine(core.Engine):
    """The frozen Engine (loading, encode, forward, capture, generate, score_candidates) with span writes added."""

    def __init__(self, phase, results):
        results.mkdir(parents=True, exist_ok=True)
        core.previous.RESULTS = results
        super().__init__('qwen4b', phase)
        self.format_name = 'fewshot'

    @contextlib.contextmanager
    def patch(self, spec=None, stats=None):
        if spec is None or spec[3] not in ('span_replace', 'span_add'):
            with core.Engine.patch(self, spec, stats):  # the frozen single-position hook
                yield
            return
        layer, positions, vectors, rule = spec
        index = torch.tensor(positions, dtype=torch.long)

        def hook(module, args, output):
            assert isinstance(output, torch.Tensor) and output.ndim == 3 and max(positions) < output.shape[1]
            after = output.clone()
            idx = index.to(output.device)
            old = output[0, idx]
            target = vectors.to(old)
            if rule == 'span_add':
                target = old + target
            after[0, idx] = target
            if stats is not None and not stats:
                others = after - output
                others[0, idx] = 0
                stats.update(layer=layer, positions=list(positions), delta_norm=float((target - old).float().norm()),
                             write_error=float((after[0, idx] - target).abs().max()),
                             other_positions_max_change=float(others.abs().max()))
            return after
        handle = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def capture_at(self, encoded, positions):
        """Residual stream after every block at the given positions: tensor [blocks, positions, width] (bfloat16, GPU)."""
        result, handles = {}, []
        for layer, module in enumerate(self.model.model.layers):
            def hook(module, args, output, layer=layer):
                result[layer] = output[0, positions].detach().clone()
            handles.append(module.register_forward_hook(hook))
        try:
            self.forward(encoded['prompt_ids'])
        finally:
            for h in handles:
                h.remove()
        return torch.stack([result[i] for i in range(len(handles))])

    def segments(self, encoded):
        """Token positions of the first-step sentence: A (lead words), B (between the operands), C (after the second
        operand, through the end of the first step). The operand tokens themselves are in none of them."""
        serial = encoded['serialized_prompt']
        enc = self.tok(serial, add_special_tokens=False, return_offsets_mapping=True)
        assert enc['input_ids'] == encoded['prompt_ids']
        start = serial.rindex(encoded['prompt'])
        off = enc['offset_mapping']
        first = [i for i, (a, b) in enumerate(off) if a == start]
        a_first = [i for i, (a, b) in enumerate(off) if a == start + len(LEAD)]
        assert len(first) == 1 and len(a_first) == 1 and encoded['prompt'].startswith(LEAD)
        p = encoded['positions']
        seg = dict(A=list(range(first[0], a_first[0])), B=list(range(p['a_end'] + 1, p['b_end'])),
                   C=list(range(p['b_end'] + 1, p['first_stage_end'] + 1)))
        return seg

    def span_pairs(self, rec, src, span):
        """Aligned (recipient position, source position) lists for a span, segment by segment."""
        names = ['C'] if span == 'post_operand' else ['A', 'B', 'C']
        rs, ss = self.segments(rec), self.segments(src)
        rpos, spos = [], []
        for n in names:
            assert len(rs[n]) == len(ss[n]), (n, rs[n], ss[n])
            rpos += rs[n]
            spos += ss[n]
        return rpos, spos


def write_pair(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp, 'wt') as fh:
        for r in records:
            fh.write(json.dumps(r) + '\n')
    tmp.rename(path)


def gen_record(engine, key, row, encoded, spec=None, forced=None, **fields):
    t0 = time.perf_counter()
    out = engine.generate(encoded, spec, forced)
    return dict(key=key, id=row['id'], group=row['group'], **fields, output_ids=out['output_ids'], text=out['text'],
                answer=out['answer'], truncated=out['truncated'], write=out['write'],
                seconds=round(time.perf_counter() - t0, 4))


# ======================================================================================== first stage checks
def instrumentation(engine, rows):
    """The frozen instrumentation on the first four rows, plus the span hook: writing a recipient's own vectors over
    either span, or adding zero over it, changes no logit by more than 1e-5, and a donor span write changes only its
    positions."""
    frozen_pass = core.instrumentation(engine, rows)
    checks = []
    for row in rows[:4]:
        enc = engine.encode(*prompt(row['recipient'], row['family']))
        base = engine.forward(enc['prompt_ids'])
        for span in SPANS:
            rpos, _ = engine.span_pairs(enc, enc, span)
            own = engine.capture_at(enc, rpos)
            for layer in [0, 18, 35]:
                for rule, vec in [('span_replace', own[layer]), ('span_add', torch.zeros_like(own[layer]))]:
                    delta = float((engine.forward(enc['prompt_ids'], spec=(layer, rpos, vec, rule)) - base).abs().max())
                    checks.append(dict(id=row['id'], span=span, layer=layer, rule=rule, max_logit_abs=delta, pass_check=delta <= 1e-5))
    row = rows[0]
    rec = engine.encode(*prompt(row['recipient'], row['family']))
    don = engine.encode(*prompt(row['donor'], row['family']))
    rpos, spos = engine.span_pairs(rec, don, 'first_step')
    vec = engine.capture_at(don, spos)[18]
    stats = {}
    base = engine.forward(rec['prompt_ids'])
    effect = engine.forward(rec['prompt_ids'], spec=(18, rpos, vec, 'span_replace'), stats=stats)
    stats['max_logit_change'] = float((effect - base).abs().max())
    positive = stats['delta_norm'] > 0 and stats['write_error'] == 0 and stats['other_positions_max_change'] == 0 and stats['max_logit_change'] > 0
    return dict(frozen_instrumentation=frozen_pass, span_checks=checks, span_positive_write=stats,
                pass_check=bool(frozen_pass and all(c['pass_check'] for c in checks) and positive))


def reproduction(engine, rows, reference, vectors):
    """Regenerates the saved Study 3 records of donor group 0 (box wording) and compares them token for token; the
    captured donor vectors are compared with the saved ones (byte equality and largest absolute difference)."""
    by_id = {r['id']: r for r in rows}
    checks = []
    caps = {}
    for ref in reference['records']:
        row = by_id[ref['id']]
        if ref['kind'] == 'baseline':
            side = 'donor' if ref['condition'] == 'donor_full' else 'recipient'
            enc = engine.encode(*prompt(row[side], row['family']))
            out = engine.generate(enc)
            checks.append(dict(key=ref['key'], prompt_equal=enc['prompt_ids'] == ref['prompt_ids'],
                               tokens_equal=out['output_ids'] == ref['output_ids'], answer=out['answer'], saved_answer=ref['answer']))
            continue
        rec = engine.encode(*prompt(row['recipient'], row['family']))
        if ref['condition'] == 'unpatched':
            out = engine.generate(rec)
            checks.append(dict(key=ref['key'], prompt_equal=rec['prompt_ids'] == ref['prompt_ids'],
                               tokens_equal=out['output_ids'] == ref['output_ids'], answer=out['answer'], saved_answer=ref['answer']))
            continue
        if row['id'] not in caps:
            caps[row['id']] = engine.capture(engine.encode(*prompt(row['donor'], row['family'])))
        vec = caps[row['id']][ref['layer'], ROLES.index(ref['role'])]
        saved = vectors[ref['key']]
        out = engine.generate(rec, (ref['layer'], rec['positions'][ref['role']], vec, 'replace'))
        checks.append(dict(key=ref['key'], prompt_equal=rec['prompt_ids'] == ref['prompt_ids'],
                           tokens_equal=out['output_ids'] == ref['output_ids'], answer=out['answer'], saved_answer=ref['answer'],
                           vector_bytes_equal=digest(vec) == ref['source_hash'],
                           vector_max_abs_difference=float((vec.float() - saved.float()).abs().max()),
                           vector_max_abs=float(saved.float().abs().max())))
    passed = all(c['prompt_equal'] and c['tokens_equal'] for c in checks)
    return dict(rule='PROTOCOL.md section 8: every regenerated record equals the saved Study 3 record token for token',
                records=len(checks), tokens_equal=sum(c['tokens_equal'] for c in checks),
                vectors_bytes_equal=sum(c.get('vector_bytes_equal', False) for c in checks),
                vectors_checked=sum('vector_bytes_equal' in c for c in checks),
                vector_max_abs_difference=max(c.get('vector_max_abs_difference', 0.0) for c in checks),
                checks=checks, pass_check=passed)


def class_means(engine, fitting):
    """Mean residual vector after every block at the four positions for every intermediate value of the fitting set:
    tensor [values, blocks, 4, width] in float32, with the values in order."""
    values = sorted({r['z'] for r in fitting})
    index = {v: i for i, v in enumerate(values)}
    sums = None
    counts = torch.zeros(len(values))
    for n, r in enumerate(fitting, 1):
        enc = engine.encode(*prompt(r['task'], 'box'))
        cap = engine.capture(enc).float()
        if sums is None:
            sums = torch.zeros((len(values),) + tuple(cap.shape))
        sums[index[r['z']]] += cap
        counts[index[r['z']]] += 1
        if n % 200 == 0:
            print('fitting', n, '/', len(fitting), flush=True)
    return values, sums / counts[:, None, None, None], counts


# ======================================================================================== per-pair work
class Pair:
    def __init__(self, engine, row, means, values, group_donor):
        self.e, self.row = engine, row
        self.means, self.vi = means, {v: i for i, v in enumerate(values)}
        self.rec = engine.encode(*prompt(row['recipient'], row['family']))
        self.tasks = dict(donor=row['donor'], random_source=group_donor[(row['group'] + 1) % len(group_donor)],
                          **row['controls'])
        self._enc, self._span = {}, {}

    def enc(self, source):
        if source not in self._enc:
            self._enc[source] = self.e.encode(*prompt(self.tasks[source], self.row['family']))
        return self._enc[source]

    def direction(self, z_source, layer, role, alpha):
        d = self.means[self.vi[z_source], layer, ROLES.index(role)] - self.means[self.vi[self.row['recipient']['z']], layer, ROLES.index(role)]
        return alpha * d

    def span_vectors(self, source, span):
        key = (source, span)
        if key not in self._span:
            rpos, spos = self.e.span_pairs(self.rec, self.enc(source), span)
            self._span[key] = (rpos, self.e.capture_at(self.enc(source), spos))
        return self._span[key]

    def own_span(self, span):
        key = ('self', span)
        if key not in self._span:
            rpos, _ = self.e.span_pairs(self.rec, self.rec, span)
            self._span[key] = (rpos, self.e.capture_at(self.rec, rpos))
        return self._span[key]


def baselines(p):
    row = p.row
    out = [gen_record(p.e, f'{row["id"]}/baseline/recipient', row, p.rec, kind='baseline', condition='recipient_full'),
           gen_record(p.e, f'{row["id"]}/baseline/donor', row, p.enc('donor'), kind='baseline', condition='donor_full')]
    return out


def direction_record(p, layer, role, alpha, source='donor', noise=False):
    row = p.row
    z = p.tasks[source]['z']
    vec = p.direction(z, layer, role, alpha)
    if noise:
        key = f'{row["id"]}/direction-noise/{layer}/{role}/{alpha}'
        vec = seeded_noise(key, tuple(vec.shape), float(vec.norm()))
        cond = 'noise'
    else:
        key = f'{row["id"]}/direction/{layer}/{role}/{alpha}/{source}'
        cond = source
    spec = (layer, p.rec['positions'][role], vec, 'add')
    return gen_record(p.e, key, row, p.rec, spec, kind='direction', layer=layer, role=role, alpha=alpha,
                      condition=cond, source_z=z, direction_norm=float(vec.norm()))


def direction_scores(p, layer, role, alpha=1):
    row = p.row
    cf, normal = row['predictions']['counterfactual'], row['recipient']['y']
    vec = p.direction(row['donor']['z'], layer, role, alpha)
    t0 = time.perf_counter()
    s = p.e.score_candidates(p.rec, [cf, normal], (layer, p.rec['positions'][role], vec, 'add'))
    return dict(key=f'{row["id"]}/direction-score/{layer}/{role}/{alpha}', id=row['id'], group=row['group'], kind='direction_score',
                layer=layer, role=role, alpha=alpha, counterfactual_log_probability=s[cf]['log_probability'],
                recipient_log_probability=s[normal]['log_probability'], seconds=round(time.perf_counter() - t0, 4))


def unpatched_scores(p):
    row = p.row
    cf, normal = row['predictions']['counterfactual'], row['recipient']['y']
    s = p.e.score_candidates(p.rec, [cf, normal])
    return dict(key=f'{row["id"]}/unpatched-score', id=row['id'], group=row['group'], kind='unpatched_score',
                counterfactual_log_probability=s[cf]['log_probability'], recipient_log_probability=s[normal]['log_probability'])


def span_record(p, layer, span, source='donor', noise=False):
    row = p.row
    if noise:
        rpos, dv = p.span_vectors('donor', span)
        _, own = p.own_span(span)
        key = f'{row["id"]}/span-noise/{layer}/{span}'
        diff = (dv[layer].float() - own[layer].float()).cpu()
        vec = torch.stack([seeded_noise(f'{key}/{i}', tuple(diff[i].shape), float(diff[i].norm())) for i in range(len(rpos))])
        spec = (layer, rpos, vec, 'span_add')
        return gen_record(p.e, key, row, p.rec, spec, kind='span', layer=layer, span=span, condition='noise', positions=rpos)
    rpos, vec = p.span_vectors(source, span)
    key = f'{row["id"]}/span/{layer}/{span}/{source}'
    return gen_record(p.e, key, row, p.rec, (layer, rpos, vec[layer], 'span_replace'), kind='span', layer=layer, span=span,
                      condition=source, source_z=p.tasks[source]['z'], positions=rpos)


def self_span_check(p, layer, span, reference_ids):
    rpos, own = p.own_span(span)
    r = gen_record(p.e, f'{p.row["id"]}/span-self/{layer}/{span}', p.row, p.rec, (layer, rpos, own[layer], 'span_replace'),
                   kind='span', layer=layer, span=span, condition='self', positions=rpos)
    assert r['output_ids'] == reference_ids, 'span self write changed the output'
    return r


def run_pair(p, plan, n_layers):
    """plan: dict with the lists of work for this stage (see modal_app.py)."""
    out = baselines(p)
    base_ids = out[0]['output_ids']
    if plan.get('self_check'):
        for layer, span in plan['self_check']:
            out.append(self_span_check(p, layer, span, base_ids))
    for alpha in plan.get('direction_sweep_alphas', []):
        for layer in range(n_layers):
            for role in plan.get('direction_sweep_roles', ROLES):
                out.append(direction_record(p, layer, role, alpha))
    if plan.get('score_sweep_roles'):
        out.append(unpatched_scores(p))
        for layer in range(n_layers):
            for role in plan['score_sweep_roles']:
                out.append(direction_scores(p, layer, role))
    for span in plan.get('span_sweep', []):
        for layer in range(n_layers):
            out.append(span_record(p, layer, span))
    for site in plan.get('direction_sites', []):
        layer, role, alpha = site['layer'], site['role'], site['alpha']
        if not site.get('in_sweep'):
            out.append(direction_record(p, layer, role, alpha))
        out.append(direction_record(p, layer, role, alpha, noise=True))
        out.append(direction_record(p, layer, role, alpha, source='same_leading_different_value'))
        if role in ('first_stage_end', 'prompt_end'):
            if not any(r['kind'] == 'unpatched_score' for r in out):
                out.append(unpatched_scores(p))
            out.append(direction_scores(p, layer, role, alpha))
    for site in plan.get('span_sites', []):
        layer, span = site['layer'], site['span']
        out.append(span_record(p, layer, span, noise=True))
        for source in SPAN_SOURCES:
            out.append(span_record(p, layer, span, source=source))
    return out
