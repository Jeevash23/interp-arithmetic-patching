"""Bounded format development, separate from the frozen original prompt attempt."""
import json
from pathlib import Path
import run
from tasks import HERE, dump, prompt, sha

FROOT = HERE/'format-pilot'
SYSTEM = 'You are a calculator. Reply with exactly one integer. Do not show steps, equations, explanations, labels, or punctuation.'
EXAMPLES = [
    ('A box starts with 3 counters. Add 4 to finish the first step. Then multiply that result by 2. Return only the final integer, with no explanation.','14'),
    ('A box starts with 17 counters. Subtract 5 to finish the first step. Then add 6 to that result. Return only the final integer, with no explanation.','18'),
    ('A box starts with 24 counters. Add 3 to finish the first step. Then subtract 4 from that result. Return only the final integer, with no explanation.','23')]


class FormatEngine(run.Engine):
    def __init__(self, phase, format_name='system'):
        self.format_name = format_name
        super().__init__(phase)

    def encode(self, text, spans=None):
        messages = [dict(role='system',content=SYSTEM)]
        if self.format_name == 'fewshot':
            for inp,out in EXAMPLES:
                messages.extend([dict(role='user',content=inp),dict(role='assistant',content=out)])
        messages.append(dict(role='user',content=text))
        serial = self.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded = self.tok(serial,add_special_tokens=False,return_offsets_mapping=True)
        ids = encoded['input_ids']
        assert ids == self.tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        start = serial.rindex(text)
        positions = {'prompt_end':len(ids)-1}
        for name,end in (spans or {}).items():
            matches=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if start < b <= start+end]
            positions[name]=max(matches)
            assert encoded['offset_mapping'][positions[name]][1]==start+end
        return dict(prompt=text,serialized_prompt=serial,prompt_ids=ids,positions=positions)


def expression(t,kind='full',z=None):
    op={'add':'+','sub':'-','mul':'*'}
    if kind=='counterfactual':
        expr=f"{z} {op[t['remaining']]} {t['k']}"
    else:
        expr=f"({t['a']} {op[t['first']]} {t['b']}) {op[t['remaining']]} {t['k']}"
    return f'Compute {expr}. Return only the final integer.',{}


def main():
    run.freeze()
    manifest = dict(files={str(p.relative_to(HERE)):sha(p) for p in [HERE/'FORMAT-PILOT.md',HERE/'prompt_formats.py',HERE/'data/development.jsonl']})
    fp=FROOT/'freeze.json'
    if fp.exists():
        assert json.loads(fp.read_text())==manifest
    else:
        dump(fp,manifest)
    run.RESULTS=FROOT/'results'
    engine=FormatEngine('formats')
    rows=[json.loads(s) for s in (HERE/'data/development.jsonl').read_text().splitlines() if json.loads(s)['group']<2]
    j=run.Journal('baselines')
    reuse={tuple(r['prompt_ids']):r for r in j.rows.values()}
    for fmt in ['system','fewshot','symbolic']:
        engine.format_name=fmt
        for row in rows:
            for condition,side,kind in [('donor_full','donor','full'),('recipient_full','recipient','full'),('textual_counterfactual','recipient','counterfactual')]:
                key=f'{fmt}/{row["id"]}/{condition}'
                if key in j.rows:continue
                z=row['donor']['z'] if kind=='counterfactual' else None
                pr=expression(row[side],kind,z) if fmt=='symbolic' else prompt(row[side],row['family'],kind,z)
                enc=engine.encode(*pr)
                old=reuse.get(tuple(enc['prompt_ids']))
                out=engine.generate(enc) if old is None else {k:old[k] for k in ['output_ids','text','answer','truncated','write','forced_output_ids']}
                target=row['predictions']['counterfactual'] if z is not None else row[side]['y']
                saved=j.put(key,dict(format=fmt,id=row['id'],condition=condition,target=target,correct=out['answer']==target,reused_from=old['key'] if old else None,**enc,**out))
                reuse[tuple(enc['prompt_ids'])]=saved
            print('Format check:',fmt,row['id'],flush=True)
    summary={}
    for fmt in ['system','fewshot','symbolic']:
        summary[fmt]={c:dict(n=len(rr),correct=sum(r['correct'] for r in rr),invalid=sum(r['answer'] is None for r in rr),truncated=sum(r['truncated'] for r in rr)) for c in ['donor_full','recipient_full','textual_counterfactual'] if (rr:=[r for r in j.rows.values() if r['format']==fmt and r['condition']==c])}
    passed=[fmt for fmt in ['system','fewshot','symbolic'] if all(s['correct']/s['n']>=.9 for s in summary[fmt].values())]
    result=dict(conditions=summary,selected_first_passing=passed[0] if passed else None,reserve_evaluated=False,localization_run=False)
    dump(run.RESULTS/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
