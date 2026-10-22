"""Non-visual artifact checks; call only after inspecting every rendered page."""
from pathlib import Path
import json,re,hashlib,argparse
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--visually-reviewed-pages',required=True,type=int);args=a.parse_args()
pdf=ROOT/'01-Papers/reasoning-interventions-report.pdf';reader=PdfReader(pdf);issues=[]
if len(reader.pages)!=args.visually_reviewed_pages:issues.append('visual_review_count')
texts=[p.extract_text() or '' for p in reader.pages]
for i,t in enumerate(texts):
 if len(t)<70:issues.append(f'page_{i+1}_empty')
 if '\ufffd' in t or '\x00' in t:issues.append(f'page_{i+1}_replacement_character')
for p in [ROOT/'00-START-HERE.md',ROOT/'README.md',ROOT/'01-Papers/manuscript.md',ROOT/'05-Notes/REPRODUCING.md',ROOT/'05-Notes/CLAIM-TO-EVIDENCE.md',ROOT/'05-Notes/HTML-COMPANION-HANDOFF.md']:
 for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  if target.startswith(('https://','http://','#')):continue
  target=target.split('#')[0]
  if target and not (p.parent/target).exists():issues.append(str(p.relative_to(ROOT))+' missing '+target)
for name in ['01-assay-comparison','02-control-comparison','03-observed-trajectory','04-completion-and-cost','05-donor-semantics']:
 for ext in ['png','svg']:
  if not (ROOT/'04-Results/figures'/f'{name}.{ext}').exists():issues.append('missing figure '+name+'.'+ext)
for word in ['TODO','TBD','PLACEHOLDER','INSERT RESULTS']:
 if word in '\n'.join(texts):issues.append('unfinished_marker_'+word)
result=dict(passed=not issues,issues=issues,pages=len(reader.pages),visually_reviewed_pages=args.visually_reviewed_pages,page_text_characters=[len(t) for t in texts],pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),manuscript_sha256=hashlib.sha256((ROOT/'01-Papers/manuscript.md').read_bytes()).hexdigest(),note='Numeric source checks are in validation-v1/checks.json. Page count is an explicit record of completed visual review, not an automated visual assessment.')
(ROOT/'04-Results/analysis/artifact-qa.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if issues:raise SystemExit(1)
