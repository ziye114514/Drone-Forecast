"""Instrument the actual frozen count-adapter threshold; no math replacement."""
from pathlib import Path
import argparse,ast,copy,datetime,hashlib,sys
import numpy as np
import study57 as current
sys.path[:0]=[str(current.ROOT/'research/stage8'),str(current.ROOT/'research/stage9'),str(current.ROOT/'research/stage55')]
import polarity
from recovery import RecoveryFrontend
from roi55 import GeometryROI
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=current.OUT;PROTOCOL=OUT/'density_probe_supplement.json'

class Probe(ast.NodeTransformer):
    def __init__(self):self.changed=0
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='threshold':
            self.changed+=1
            extra=ast.parse("self.density_probe=dict(threshold=float(threshold),positive_bins=int(len(positive_bins)))").body[0]
            return [node,extra]
        return self.generic_visit(node)

def instrument():
    tree=copy.deepcopy(polarity._tree);probe=Probe();tree=ast.fix_missing_locations(probe.visit(tree));assert probe.changed==1
    return tree,ast.unparse(tree)

def freeze():
    current.verify();assert not PROTOCOL.exists();tree,source=instrument()
    inputs={name:current.sha(current.previous.OUT/'clear6'/name) for name in ['observations_forecasts.jsonl','rotor_ticks.jsonl']}
    inputs['off/rotor_ticks.jsonl']=current.sha(current.previous.OUT/'off_clear6/rotor_ticks.jsonl')
    current.save(PROTOCOL,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),
        instrumented_source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),input_hashes=inputs,
        case='clear6',start=22.,duration=2.,modes=['full','clipped'],
        design='Actual8 transformed detect AST: add only self.density_probe assignment after original threshold assignment. No numeric expressions changed. Two fresh processes each read actual RAW10; same past body hints as55. Compare every complete current geometry observation exactly against55 off/hard ROI. No frequency, future labels, model fitting or parameter selection.',
        limits='This isolates the threshold values under the same old hard-crop intervention, not the causal contribution of threshold vs boundary clipping to final forecast error. One known regression clip; not a new generalization benchmark.'))
    (OUT/'derived_density_probe.py').write_text(source,encoding='utf-8');print('Actual threshold AST probe frozen',flush=True)

def verify():
    current.verify();p=current.load(PROTOCOL)
    assert p['helper_sha256']==current.sha(Path(__file__)) and p['instrumented_source_sha256']==hashlib.sha256(instrument()[1].encode('utf-8')).hexdigest()
    for n,v in p['input_hashes'].items():
        path=current.previous.OUT/'off_clear6/rotor_ticks.jsonl' if n.startswith('off/') else current.previous.OUT/'clear6'/n
        assert current.sha(path)==v

def run(mode):
    verify();path=OUT/f'density_probe_{mode}.json';assert not path.exists()
    tree,source=instrument();namespace=dict(polarity._reused_detect.__globals__);exec(compile(tree,'<57 actual8 threshold probe>','exec'),namespace)
    polarity._reused_detect=namespace['detect']
    parent=current.rows(current.previous.OUT/'clear6/observations_forecasts.jsonl');hints={}
    for r in parent:
        for t in [r['t']-.01,r['t']]:hints[round(t*1e6)]=r['body_hint_sent']
    reference=current.rows(current.previous.OUT/('off_clear6' if mode=='full' else 'clear6')/'rotor_ticks.jsonl')
    cfg=current.CASES['clear6'];reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();front=RecoveryFrontend(w,h,'neighbor10');roi=GeometryROI(w,h,'body');result=[]
    for index,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
        if mode=='clipped':selected,d=roi.apply(e,hints[round(t*1e6)],t)
        else:selected=e;d=dict(mode='global')
        obs,_=front.detect(selected,t);expected=reference[index];assert round(t,8)==round(expected['t'],8)
        assert obs==expected['observation'],(mode,t,'instrumented geometry not exact55')
        result.append(dict(t=t,raw_events=len(e),geometry_events=len(selected),roi_mode=d['mode'],**front.density_probe))
    assert len(result)==200
    current.save(path,dict(table=result,geometry_exact55=True,native10=200,labels_read=False,source_unchanged=True));print(f'{mode}:200 actual RAW geometry observations exact55',flush=True)

def report():
    verify();a=current.load(OUT/'density_probe_full.json');b=current.load(OUT/'density_probe_clipped.json')
    assert a['geometry_exact55'] and b['geometry_exact55'];rr=[]
    for x,y in zip(a['table'],b['table']):
        assert x['t']==y['t'];rr.append(dict(t=x['t'],full_threshold=x['threshold'],clipped_threshold=y['threshold'],threshold_ratio=y['threshold']/x['threshold'],
            full_positive_bins=x['positive_bins'],clipped_positive_bins=y['positive_bins'],active_roi=y['roi_mode']=='body_roi'))
    active=[r for r in rr if r['active_roi']]
    summary=dict(native10=200,active_roi_ticks=len(active),changed_threshold_ticks=sum(r['full_threshold']!=r['clipped_threshold'] for r in rr),
        active_median_threshold_ratio=float(np.median([r['threshold_ratio'] for r in active])),maximum_threshold_ratio=max(r['threshold_ratio'] for r in rr),
        full_threshold_median=float(np.median([r['full_threshold'] for r in rr])),clipped_threshold_median=float(np.median([r['clipped_threshold'] for r in rr])),
        full_positive_bins_median=float(np.median([r['full_positive_bins'] for r in rr])),clipped_positive_bins_median=float(np.median([r['clipped_positive_bins'] for r in rr])))
    current.save(OUT/'density_probe_analysis.json',dict(summary=summary,table=rr,limits=current.load(PROTOCOL)['limits']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.5,4),layout='constrained');t=[r['t'] for r in rr]
    for field,label,color in [('full','Full global','#277da1'),('clipped','Hard ROI55','#e76f51')]:
        axes[0].plot(t,[r[field+'_threshold'] for r in rr],label=label,color=color,lw=1)
        axes[1].plot(t,[r[field+'_positive_bins'] for r in rr],label=label,color=color,lw=1)
    for ax in axes:ax.set_xlabel('RAW source time (s)');ax.legend(fontsize=8)
    axes[0].set_ylabel('Actual density threshold');axes[1].set_ylabel('Nonzero blurred spatial bins');fig.suptitle('Same clear6 RAW; exact original55 geometry reproduced, one logging assignment only',fontsize=11)
    fig.savefig(OUT/'density_threshold_probe.png',dpi=150);plt.close(fig);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','report']);p.add_argument('--mode',choices=['full','clipped']);a=p.parse_args()
    if a.action=='run':run(a.mode)
    else:globals()[a.action]()
