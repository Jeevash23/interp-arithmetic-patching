from pathlib import Path
import argparse,json,re,collections
from ai04.common import ROOT,readj,writej
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args();d=ROOT/'04-Results'/a.run
rows=[json.loads(l) for l in (d/'records.jsonl').read_text().splitlines()];gs={g['id']:g for g in readj(ROOT/readj(d/'freeze.json')['config']['data'])};records=[]
for r in rows:
 if r.get('kind')!='intervention' or not r.get('available'):continue
 g=gs[r['group']];start=r['timing']['char_start'];suffix=r['text'][start:];m=re.match(r'\s*(-?\d+)',suffix);first=int(m.group(1)) if m else None
 zr=g['recipient']['z'];zd=g['donor']['z'];original_later=re.search(r'(?<!\d)'+str(zr)+r'(?!\d)',suffix[m.end():] if m else suffix)
 returned=r['parsed']['final']==g['recipient']['y'] and r['stop_reason']=='eos'
 records.append(dict(key=r['key'],group=r['group'],site=r['site'],condition=r['condition'],first_numeral=first,first_digit_copied=first is not None and str(first)[0]==str(zd)[0],complete_donor_numeral=first==zd,changed_numeral=first is not None and first!=zr,returned_recipient=returned,changed_then_original_mentioned_and_returned=first is not None and first!=zr and bool(original_later) and returned,explicit_error_word=bool(re.search(r'\b(mistake|error|incorrect|wrong|miscalculat\w*)\b',suffix,re.I)),completed=r['stop_reason']=='eos'))
summary={}
for site,cond in sorted({(r['site'],r['condition']) for r in records}):
 rr=[r for r in records if r['site']==site and r['condition']==cond];fields=['first_digit_copied','complete_donor_numeral','changed_numeral','returned_recipient','changed_then_original_mentioned_and_returned','explicit_error_word'];summary[site+'/'+cond]=dict(n=len(rr),**{f:sum(r[f] for r in rr) for f in fields})
writej(ROOT/a.output,dict(run=a.run,operational_note='Changed first numeral + later mention + recipient final is a trajectory signature, not by itself proof of recomputation or belief; explicit error words are separately counted and require contextual review.',summary=summary,records=records));print(json.dumps(summary,indent=2))
