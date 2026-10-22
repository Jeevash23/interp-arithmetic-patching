"""Locate an explicit first-stage equation without inserting an expected continuation."""
import re

def first_equation_match(text,task):
    # Match the original operands/operator; capture the model's result, even when wrong.
    a,b=task['a'],task['b']
    symbol=r'(?:\+|plus)' if task['first']=='add' else r'(?:-|−|minus)'
    pattern=rf'(?<!\d){a}\s*{symbol}\s*{b}(?!\d)\s*(?:=|equals?|is|gives|makes)\s*(-?\d+)(?!\d)'
    cleaned=text.replace('**','  ')
    # Focus the locator on thinking, excluding the post-thinking answer labels.
    stop=cleaned.find('</think>')
    return re.search(pattern,cleaned[:stop] if stop>=0 else cleaned,re.I)

def equation_cuts(tokenizer,record,task):
    ids=record['output_ids'];text=tokenizer.decode(ids,skip_special_tokens=False)
    m=first_equation_match(text,task)
    if not m:return {},{'available':False,'reason':'no explicit matching operand/operator equation'}
    boundaries=[len(tokenizer.decode(ids[:i],skip_special_tokens=False)) for i in range(len(ids)+1)]
    start,end=m.span(1)
    before=max(i for i,n in enumerate(boundaries) if n<=start)
    after=next(i for i,n in enumerate(boundaries) if n>=end)
    prefix=record['prompt_ids']
    z=task['z'];earlier=len(re.findall(r'(?<!\d)'+str(z)+r'(?!\d)',text[:start]))
    info={'available':True,'matched_text':m.group(0),'character_span':list(m.span()),'emitted_value':int(m.group(1)),'oracle_z':z,'emitted_value_correct':int(m.group(1))==z,'earlier_z_literal_count':earlier,'before_output_tokens':before,'after_output_tokens':after,'locator':'first explicit equation with original a, first operator and b'}
    return {'pre_equation_result':prefix+ids[:before],'post_equation_result':prefix+ids[:after]},info
