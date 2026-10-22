"""The three candidate models of Study 4 in their fixed order, their pinned revisions, and the model-specific
prompt and parser details. Standard library only, so the tokenizer check and the analysis can import it."""
import re

ORDER = ['phi4mini', 'smollm3', 'olmo2-1b']

CANDIDATES = {
    'phi4mini': dict(
        repository='microsoft/Phi-4-mini-instruct', revision='cfbefacb99257ffa30c83adab238a50856ac3083',
        licence='MIT', precision='fp8',
        tokenizer_files=['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                         'config.json', 'generation_config.json'],
        weight_files=['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors', 'model.safetensors.index.json'],
        system_suffix='', end_strings=['<|end|>', '<|endoftext|>']),
    'smollm3': dict(
        repository='HuggingFaceTB/SmolLM3-3B', revision='a07cc9a04f16550a088caea529712d1d335b0ac1',
        licence='Apache-2.0', precision='fp8',
        tokenizer_files=['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
                         'config.json', 'generation_config.json'],
        weight_files=['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors', 'model.safetensors.index.json'],
        # The template prints only the instruction when this marker is present, which keeps its date-stamped
        # metadata block out of the prompt; thinking is disabled through the template argument.
        system_suffix=' /system_override', end_strings=['<|im_end|>', '<|end_of_text|>']),
    'olmo2-1b': dict(
        repository='allenai/OLMo-2-0425-1B-Instruct', revision='48d788eca847d4d7548f375ad03d3c9312f6139e',
        licence='Apache-2.0', precision='bf16',
        tokenizer_files=['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json',
                         'generation_config.json'],
        weight_files=['model.safetensors'],
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
