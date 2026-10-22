"""Prepare inspectable matched development translations without model evaluation."""
import json
import sys
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from value_transfer.tasks import load_data
from value_transfer.multilingual import translated_prompt

def main():
    rows=load_data('development');output=[]
    for row in rows:
        for language in ['en','es','ta']:
            for side in ['donor','recipient']:
                t=row[side];text=translated_prompt(t,row['family'],language)
                assert re.findall(r'\d+',text)==[str(t[x]) for x in ['a','b','k']]
                output.append({'id':row['id'],'language':language,'side':side,'task':t,'family':row['family'],'prompt':text,'review_status':'agent_reviewed_draft_not_independently_human_verified' if language!='en' else 'original_English','translation_note':'Tamil uses stones for counting pieces; mathematically equivalent adaptation' if language=='ta' else None})
    path=ROOT/'data/multilingual-development-drafts.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    print(f'Prepared {len(output)} prompt records; numeric placeholders verified; no model evaluation.')

if __name__=='__main__':main()
