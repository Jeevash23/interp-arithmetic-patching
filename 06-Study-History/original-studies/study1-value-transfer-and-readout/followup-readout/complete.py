"""Verify and archive the delivered follow-up without changing original artifacts."""
import hashlib,json,re,shutil
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    manifest_path=HERE/'completion-manifest.json';archive=HERE/'source-at-completion'
    assert not manifest_path.exists() and not archive.exists(),'Completion already archived'
    parent=ROOT/'results/v3/completion-manifest.json';parent_data=json.loads(parent.read_text())
    changed=[r['path'] for r in parent_data['files'] if digest(ROOT/r['path'])!=r['sha256']]
    assert not changed,changed
    for fp in [HERE/'freeze.json',HERE/'checkpoint-replication/freeze.json']:
        f=json.loads(fp.read_text());assert all(digest(ROOT/p)==h for p,h in f['files'].items())
    old=json.loads((HERE/'source-before-budget-fix/freeze.json').read_text())
    for p,h in old['files'].items():
        path=HERE/'source-before-budget-fix/run_replication.py' if p=='followup-readout/run_replication.py' else ROOT/p
        assert digest(path)==h,p
    cm=json.loads((HERE/'checkpoint-replication/model-manifest.json').read_text())
    assert all(digest(HERE/'checkpoint-replication/model'/r['name'])==r['sha256'] for r in cm['files'])
    main_audit=json.loads((HERE/'results/analysis/audit.json').read_text());cp=json.loads((HERE/'checkpoint-replication/results/analysis.json').read_text())
    assert main_audit['pass'] and cp['audit_pass']
    qa=json.loads((HERE/'results/pdf-content-check.json').read_text());assert qa['replacement_glyphs']==0 and qa['required_results_present'] and qa['empty_pages']==0
    selected=[]
    for p in HERE.rglob('*'):
        if p.is_file() and not any(x in p.relative_to(HERE).parts for x in ['model','.cache','__pycache__','tmp','source-at-completion']):selected.append(p)
    missing=[];links=0
    for p in selected:
        if p.suffix!='.md' or any(part.startswith('source-before') for part in p.parts):continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#')[0].strip('<>');links+=1
            if not (p.parent/target).exists():missing.append([str(p),target])
    assert not missing,missing
    sources=[p for p in selected if p.suffix in ['.py','.md'] and 'results' not in p.relative_to(HERE).parts and not any(part.startswith('source-before') for part in p.relative_to(HERE).parts)]
    sources += [HERE/n for n in ['config.json','data.jsonl','freeze.json','checkpoint-replication/freeze.json','checkpoint-replication/model-manifest.json']]
    archive.mkdir()
    for p in sources:
        dst=archive/p.relative_to(HERE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);assert digest(p)==digest(dst)
    selected+=list(p for p in archive.rglob('*') if p.is_file())
    manifest={'completed_utc':datetime.now(timezone.utc).isoformat(),'kind':'completion archive, not a pre-outcome freeze','main_replication_audit_pass':main_audit['pass'],'main_replication_audit_checks':len(main_audit['checks']),'second_checkpoint_mechanism_qualified':cp['complete']['mechanism_evaluated'],'second_checkpoint_status':cp['complete'],'original_validation_evaluated':False,'original_files_verified_unchanged':len(parent_data['files']),'parent_completion_manifest':{'path':str(parent.relative_to(ROOT)),'sha256':digest(parent)},'model_files_verified':len(cm['files']),'local_links_verified':links,'pdf_pages':qa['pages'],'source_archive_files':len(sources),'exclusions':'Model weights, caches, temporary render files and this manifest itself are excluded; model file hashes remain in the verified model manifest.','files':[{'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(set(selected))]}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['main_replication_audit_checks','second_checkpoint_mechanism_qualified','original_files_verified_unchanged','model_files_verified','local_links_verified','pdf_pages','source_archive_files']}))

if __name__=='__main__':main()
