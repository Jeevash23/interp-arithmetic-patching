"""Tokenizer check of the three candidates (PROTOCOL.md section 3), from the tokenizer files alone.

For each candidate, with the tokenizer files at the pinned revision (tokenizers/<candidate>/):
  1. every digit 0-9 is one token;
  2. every integer 0-999 is one token on its own, as the whole assistant answer right after the chat template's
     generation prompt, and inside the problem sentence (where the number must be one token whose offsets cover
     exactly the number);
  3. every operand, intermediate, answer, counterfactual and outcome-defining prediction (leading-digit splices,
     operand substitutions) in the three splits lies in 0-999, so it is one token;
  4. in every prompt of the three splits, the first and second operands are single tokens at their positions.
Writes results/tokenizer-check.json. Needs the tokenizers library of the restored AI-01 environment; no weights.

  python -B code/tokenizer_check.py
"""
import hashlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'code'))
sys.path.insert(0, str(STUDY / 'code/frozen'))
from candidates import CANDIDATES, ORDER, messages  # noqa: E402
from tasks import prompt, calc  # noqa: E402  frozen, standard library only
from transformers import AutoTokenizer  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rd(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def split_numbers(rows):
    out = set()
    for row in rows:
        d, r = row['donor'], row['recipient']
        for t in [d, r, *row['controls'].values()]:
            out |= {t['a'], t['b'], t['k'], t['z'], t['y']}
        g = lambda z: calc(r['remaining'], z, r['k'])
        out |= {row['predictions']['counterfactual'], row['predictions']['recipient_answer'], row['predictions']['donor_answer'],
                *row['digit_alternatives'].values(), g(calc(r['first'], d['a'], r['b'])), g(calc(r['first'], r['a'], d['b']))}
    return out


def check(name, splits):
    cfg = CANDIDATES[name]
    folder = STUDY / 'tokenizers' / name
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    one = lambda s: len(tok.encode(s, add_special_tokens=False)) == 1
    res = dict(repository=cfg['repository'], revision=cfg['revision'], tokenizer_class=type(tok).__name__,
               files={f: sha(folder / f) for f in cfg['tokenizer_files']})
    res['digits_single'] = sum(one(str(d)) for d in range(10))
    res['standalone_0_999_single'] = sum(one(str(n)) for n in range(1000))
    # As the whole assistant answer, right after the generation prompt.
    prefix = tok.apply_chat_template(messages(name, prompt(splits['calibration'][0]['recipient'])[0]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    base_ids = tok(prefix, add_special_tokens=False)['input_ids']
    answer_single, failures = 0, []
    for n in range(1000):
        ids = tok(prefix + str(n), add_special_tokens=False)['input_ids']
        if ids[:len(base_ids)] == base_ids and len(ids) == len(base_ids) + 1:
            answer_single += 1
        elif len(failures) < 5:
            failures.append(n)
    res['answer_0_999_single'] = answer_single
    res['answer_failures_first'] = failures
    res['generation_prompt_tail'] = prefix[-40:]
    # Inside the problem sentence.
    sentence_single = 0
    for n in range(1000):
        text = f'A box starts with {n} counters. Add 4 to finish the first step.'
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        start = text.index(str(n)); end = start + len(str(n))
        sentence_single += any(a == start and b == end for a, b in enc['offset_mapping'])
    res['sentence_0_999_single'] = sentence_single
    # Numbers of the splits and operand tokens of every prompt.
    numbers = set().union(*(split_numbers(rows) for rows in splits.values()))
    res['split_numbers'] = dict(count=len(numbers), min=min(numbers), max=max(numbers),
                                outside_0_999=sorted(x for x in numbers if not 0 <= x <= 999))
    operand_ok = operand_n = 0
    for rows in splits.values():
        for row in rows:
            for t in [row['donor'], row['recipient'], *row['controls'].values()]:
                text, spans = prompt(t, 'box', 'full')
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                for key, value in [('a_end', t['a']), ('b_end', t['b'])]:
                    end = spans[key]; start = end - len(str(value))
                    operand_n += 1
                    operand_ok += any(a == start and b == end for a, b in enc['offset_mapping'])
    res['prompt_operands_single'] = dict(single=operand_ok, n=operand_n)
    res['pass_check'] = (res['digits_single'] == 10 and res['standalone_0_999_single'] == 1000 and answer_single == 1000
                         and sentence_single == 1000 and not res['split_numbers']['outside_0_999'] and operand_ok == operand_n)
    return res


def main():
    splits = {s: rd(STUDY / 'data' / f'{s}.jsonl') for s in ['calibration', 'development', 'test']}
    result = dict(rule='PROTOCOL.md section 3; checked on the tokenizer files alone, before any weights were downloaded',
                  candidates={name: check(name, splits) for name in ORDER})
    result['passing_in_order'] = [n for n in ORDER if result['candidates'][n]['pass_check']]
    (STUDY / 'results').mkdir(exist_ok=True)
    (STUDY / 'results/tokenizer-check.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({n: {k: v for k, v in r.items() if k not in ('files',)} for n, r in result['candidates'].items()}, indent=1))


if __name__ == '__main__':
    main()
