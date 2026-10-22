import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260909
FORMAT = 'Show the arithmetic calculation on each of exactly two lines. Use this format, filling in the numbers and operators:\nIntermediate: <first calculation> = <intermediate integer>\nFinal: <second calculation> = <final integer>'
FINAL_FORMAT = 'Return exactly one line: Final: <integer>'

def op(a, b, name):
    return {"add": lambda: a+b, "sub": lambda: a-b, "mul": lambda: a*b}[name]()

def task(z, first, remaining, k, rng):
    b = rng.randint(1, min(9, z-1))
    a = z-b if first == "add" else z+b
    return dict(a=a, b=b, first=first, z=z, remaining=remaining, k=k, y=op(z,k,remaining))

def first_text(t, family):
    a,b=t["a"],t["b"]
    if family == "dev_box":
        return f'A box starts with {a} counters. ' + (f'Put {b} more counters in the box.' if t["first"]=="add" else f'Remove {b} counters from the box.')
    if family == "dev_score":
        return f'Your score begins at {a} points. ' + (f'First gain {b} points.' if t["first"]=="add" else f'First lose {b} points.')
    if family == "val_machine":
        return f'A machine receives the number {a}. In its first stage it ' + (f'increases this number by {b}.' if t["first"]=="add" else f'decreases this number by {b}.')
    if family == "val_ledger":
        return f'A ledger lists {a} units. The first adjustment ' + (f'adds {b} units.' if t["first"]=="add" else f'deducts {b} units.')
    raise ValueError(family)

def remaining_text(t, family):
    k=t["k"]
    verb={"add":f'add {k} to', "sub":f'subtract {k} from', "mul":f'multiply by {k}'}[t["remaining"]]
    if t["remaining"]=="mul":
        core=f'multiply the intermediate result by {k}'
    else:
        core=f'{verb} the intermediate result'
    if family.startswith("dev"):
        return f'Then {core}. What is the final result?'
    if family=="val_machine":
        return f'Its second stage must {core}. Report both stage results.'
    return f'For the second adjustment, {core}. Determine the ending value.'

def prompt(t, family, kind="full", supplied=None, distractor=None):
    if kind=="full":
        text=f'{first_text(t,family)} This completes the first step. {remaining_text(t,family)}\n{FORMAT}'
    elif kind=="first":
        text=f'{first_text(t,family)} What is the result of this step?\n{FINAL_FORMAT}'
    elif kind=="remaining":
        text=f'The intermediate result is {t["z"] if supplied is None else supplied}. {remaining_text(t,family)}\n{FINAL_FORMAT}'
    else:
        raise ValueError(kind)
    if distractor is not None:
        text=f'The unrelated reference code is {distractor}; do not use it in the calculation.\n'+text
    return text

def predictions(d,r):
    return {
        "counterfactual":op(d["z"],r["k"],r["remaining"]),
        "recipient_answer":r["y"], "donor_answer":d["y"], "donor_z":d["z"],
        "operand_transfer":op(op(d["a"],d["b"],r["first"]),r["k"],r["remaining"]),
        # Transferring the complete first expression is behaviorally identical to z here.
        "expression_transfer":op(d["z"],r["k"],r["remaining"]),
        "donor_operation":op(r["z"],d["k"],d["remaining"]),
        "donor_answer_as_z":op(d["y"],r["k"],r["remaining"]),
        "first_operand_as_z":op(d["a"],r["k"],r["remaining"]),
        "second_operand_as_z":op(d["b"],r["k"],r["remaining"]),
        "copy_first_operand":d["a"], "copy_second_operand":d["b"],
        "recipient_z":r["z"],
    }

def make_split(split, n=100):
    rng=random.Random(SEED+(0 if split=="development" else 1000))
    families=["dev_box","dev_score"] if split=="development" else ["val_machine","val_ledger"]
    rows=[]
    for i in range(n):
        group=i//5
        zd=(5 if split=="development" else 25)+group
        while True:
            zr=rng.randint(5,24) if split=="development" else rng.randint(25,44)
            fr=["add","sub"][i%2]; fd="sub" if fr=="add" else "add"
            gr=["add","mul","sub"][i%3]
            gd=rng.choice([x for x in ["add","mul","sub"] if x!=gr])
            d=task(zd,fd,gd,rng.randint(2,6),rng)
            r=task(zr,fr,gr,rng.randint(2,6),rng)
            p=predictions(d,r)
            unique=[p[k] for k in ["counterfactual","recipient_answer","donor_answer","donor_z","recipient_z","operand_transfer"]]
            if len(set(unique))==len(unique) and min(d["y"],r["y"],p["counterfactual"])>0:
                break
        dec=task(zd,fd,gd,d["k"],rng)
        while dec["a"]==d["a"]:
            dec=task(zd,fd,gd,d["k"],rng)
        other=task(zd,fd,"add" if gd!="add" else "mul",7,rng)
        z2=d["y"]-1
        if z2==zd or z2<2:
            z2=d["y"]+2
            sameanswer=task(z2,fd,"sub",2,rng)
        else:
            sameanswer=task(z2,fd,"add",1,rng)
        mismatched=task(zd+3,fd,gd,d["k"],rng)
        row={"id":f'{split}-{i:03}',"split":split,"family":families[group%2],"cluster_z":f'{split}-z-{zd}',
             "structure":f'{fd}-{gd}-{fr}-{gr}',"donor":d,"recipient":r,"predictions":p,
             "controls":{"same_z_decomposition":dec,"same_z_different_answer":other,"different_z_same_answer":sameanswer,"mismatched_z":mismatched},
             "distractor":83+(i%7)}
        rows.append(row)
    return rows

def write_data():
    manifest={"seed":SEED,"n_per_split":100,"status":"reserved_validation_not_evaluated", "splits":{}}
    for split in ["development","validation"]:
        rows=make_split(split)
        path=ROOT/f'data/{split}.jsonl'
        text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
        if path.exists() and path.read_text()!=text:
            raise RuntimeError(f'Refusing to overwrite changed dataset: {path}')
        path.write_text(text)
        manifest["splits"][split]={"sha256":hashlib.sha256(text.encode()).hexdigest(),"rows":len(rows),"families":sorted({r["family"] for r in rows}),"z_values":sorted({r["donor"]["z"] for r in rows})}
    (ROOT/'data/split-manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def load_data(split):
    return [json.loads(x) for x in (ROOT/f'data/{split}.jsonl').read_text().splitlines()]

def parse(text):
    # Anchored complete labels only. Never take the last incidental number.
    clean=text.replace('<|im_end|>','').replace('<|endoftext|>','').strip()
    calculation=r'(?:-?\d+\s*[+−–x×*\-/]\s*-?\d+\s*=\s*)?'
    final=re.findall(r'^Final:[ \t]*'+calculation+r'(-?\d+)[ \t]*$', clean, flags=re.M)
    inter=re.findall(r'^Intermediate:[ \t]*'+calculation+r'(-?\d+)[ \t]*$', clean, flags=re.M)
    return {"answer":int(final[0]) if len(final)==1 else None,
            "intermediate":int(inter[0]) if len(inter)==1 else None,
            "parser_failure":len(final)!=1,
            "format_exact":bool(re.fullmatch(r'(?:Intermediate:[ \t]*'+calculation+r'-?\d+[ \t]*\n)?Final:[ \t]*'+calculation+r'-?\d+[ \t]*',clean))}

def classify(parsed,p):
    answer=parsed["answer"]
    matches=[k for k,v in p.items() if answer is not None and answer==v]
    primary=next((k for k in ["counterfactual","recipient_answer","donor_answer","donor_z","operand_transfer","donor_operation","donor_answer_as_z","first_operand_as_z","second_operand_as_z","copy_first_operand","copy_second_operand","recipient_z"] if k in matches),"invalid" if answer is None else "other_numeric")
    return {"category":primary,"prediction_matches":matches,"success":answer==p["counterfactual"] if answer is not None else False}
