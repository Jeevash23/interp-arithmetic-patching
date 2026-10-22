"""The candidate model of Study 6, its pinned revision, and the model-specific prompt and parser details.
Study 4's candidates.py with its three candidates replaced by OLMo-2-1124-7B-Instruct (Study 6 PROTOCOL.md section 2);
SYSTEM, EXAMPLES, messages() and make_parser() are unchanged. Standard library only."""
import re

ORDER = ['olmo2-7b']

CANDIDATES = {
    'olmo2-7b': dict(
        repository='allenai/OLMo-2-1124-7B-Instruct', revision='470b1fba1ae01581f270116362ee4aa1b97f4c84',
        licence='Apache-2.0', precision='bf16',
        tokenizer_files=['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json',
                         'generation_config.json'],
        weight_files=['model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
                      'model-00003-of-00003.safetensors', 'model.safetensors.index.json'],
        system_suffix='', end_strings=['<|endoftext|>']),
}

SYSTEM = ('You are a calculator. Reply with exactly one integer. Do not show steps, equations, explanations, labels, '
          'or punctuation.')
EXAMPLES = [
    ('A box starts with 3 counters. Add 4 to finish the first step. Then multiply that result by 2. Return only the final '
     'integer, with no explanation.', '14'),
    ('A box starts with 17 counters. Subtract 5 to finish the first step. Then add 6 to that result. Return only the '
     'final integer, with no explanation.', '18'),
    ('A box starts with 24 counters. Add 3 to finish the first step. Then subtract 4 from that result. Return only the '
     'final integer, with no explanation.', '23')]


def messages(candidate, text):
    """System instruction, the three demonstrations and the problem, as in the selected Qwen3-4B configuration."""
    out = [dict(role='system', content=SYSTEM + CANDIDATES[candidate]['system_suffix'])]
    for inp, ans in EXAMPLES:
        out += [dict(role='user', content=inp), dict(role='assistant', content=ans)]
    out.append(dict(role='user', content=text))
    return out


def make_parser(candidate):
    """The frozen strict parser, removing this model's own end strings instead of Qwen's."""
    pattern = re.compile('|'.join(re.escape(s) for s in CANDIDATES[candidate]['end_strings']))

    def parse(text):
        text = pattern.sub('', text).strip()
        return int(text) if re.fullmatch(r'-?\d+', text) else None
    return parse
