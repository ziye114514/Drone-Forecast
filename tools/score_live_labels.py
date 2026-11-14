"""Score live publication+h endpoints against independent dense visible boxes.

This reads completed outputs only. It neither runs inference nor feeds labels
back to a model. Unlabelled/unknown intervals never become negative examples.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_live_pilot import reference_at,stats,read_rows

def evaluate(run,labels):
    manifest=json.loads(labels.read_text('utf-8'));records=manifest['records']
    records=sorted(records.values() if isinstance(records,dict) else records,key=lambda r:r['t'])
    publications=read_rows(run/'predictions.jsonl')
    methods=sorted({name for p in publications for name in p.get('predictions',{})})
    tables={};pairs=[]
    for h in [.1,.2,.4]:
        errors={name:[] for name in methods};missing=0;common=0
        for p in publications:
            if not p['valid']:continue
            truth=reference_at(records,p['t']+h)
            if truth is None:missing+=1;continue
            available={name:p['predictions'].get(name,{}).get(str(h)) for name in methods}
            if 'full_frame_baseline' in available and not p.get('full_frame_baseline_valid',True):available['full_frame_baseline']=None
            # Compare only identical accepted origins/targets when multiple
            # methods are supplied. Report all exclusions instead of dropping
            # a difficult case separately for one method.
            if not all(value is not None for value in available.values()):continue
            common+=1
            sample=dict(publication_t=p['t'],target_t=p['t']+h,horizon_s=h,truth=truth.tolist(),errors_px={})
            for name,value in available.items():
                error=float(np.linalg.norm(np.asarray(value)-truth));errors[name].append(error);sample['errors_px'][name]=error
            pairs.append(sample)
        tables[str(h)]=dict(common_targets=common,valid_origins_without_dense_reference=missing,
            metrics={name:stats(values) for name,values in errors.items()})
    return dict(scope='Actual publication+h visible-box center endpoint errors in native pixels; interpolation only between present labels<=150ms.',
        run=str(run),labels=str(labels),publication_slots=len(publications),valid_publications=sum(p['valid'] for p in publications),
        labelled_records=len(records),tables=tables,pairs=pairs,labels_were_inference_inputs=False,
        no_absolute_3D_or_thrust_accuracy_claim=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('Keep existing scores; choose a new output')
    result=evaluate(args.run_dir.resolve(),args.labels.resolve());out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=False))

if __name__=='__main__':main()
