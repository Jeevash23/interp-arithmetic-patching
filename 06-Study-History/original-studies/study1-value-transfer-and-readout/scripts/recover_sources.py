"""Reconstruct interrupted-screen source vectors against their original saved hashes."""
from collections import defaultdict
from run import Engine,RESULTS,Journal,torch,vector_hash,dump

def main():
    journal=Journal(RESULTS/'development/screen.jsonl');path=RESULTS/'development/screen-batched-donor-vectors.pt'
    vectors=torch.load(path,weights_only=True);missing=[r for r in journal.rows.values() if r.get('donor_source_hash') and r['key'] not in vectors]
    if not missing:print('All screening vectors are already stored.');return
    engine=Engine();groups=defaultdict(list);checks=[]
    for r in missing:groups[(tuple(r['donor_prefix_ids']),r['donor_position'])].append(r)
    for (ids,pos),rows in groups.items():
        captured=engine.capture(list(ids),sorted({r['layer'] for r in rows}),pos)
        for r in rows:
            actual=vector_hash(captured[r['layer']]);check={'key':r['key'],'original_hash':r['donor_source_hash'],'reconstructed_hash':actual,'pass':actual==r['donor_source_hash']};checks.append(check)
            if not check['pass']:dump(RESULTS/'analysis/source-reconstruction.json',{'pass':False,'checks':checks});raise RuntimeError('Source hash did not reproduce')
            vectors[r['key']]=captured[r['layer']]
    torch.save(vectors,path);dump(RESULTS/'analysis/source-reconstruction.json',{'pass':True,'reconstructed':len(checks),'checks':checks,'semantics':'reconstructed after inference from original saved prefix and pinned model; all source hashes match exactly'})
    print(f'Recovered {len(checks)} original source vectors with exact hash matches.')
if __name__=='__main__':main()
