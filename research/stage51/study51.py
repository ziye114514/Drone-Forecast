from pathlib import Path
import argparse,copy,datetime,hashlib,html,importlib.util,json,os,sys
import numpy as np
from reference51 import *
from checks51 import run as checks
sys.path.insert(0,str(ROOT/'research/stage48'));import study48 as base
from select29 import Selector,MODELS
BASE=ROOT.parent/'output';OUT=BASE/'stage51_preserved_physics_reference_20261020';CASES=base.CASES
load=base.load;save=base.save;sha=base.sha;rows=base.rows
def sources():
    result=base.sources()
    for n in ['reference51.py','checks51.py','study51.py']:result['research/stage51/'+n]=sha(Path(__file__).with_name(n))
    return result
def freeze():
    assert not OUT.exists();check=checks();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=base.inputs(),label_hashes=load(base.OUT/'protocol.json')['label_hashes'],
        cases=CASES,parameters=PARAMS,module_checks=check,
        design='Same700 current body+rotor inputs and same association as48 associated_slotfix. All six history coefficients and validity exact45. Reset-all15 experts must exactly reproduce48 associated_slotfix. Compare independent axis retention vs fixed-reference all-state retention. Save all forecasts before true sparse future scoring.',
        limits='Known development sequences, no physical truth; no new RAW runtime or CPU latency measured. Shared old frequency gate deliberately unchanged to isolate lifecycle. Current data are not blind.',
        prior_art=['https://arxiv.org/abs/2604.18289','https://arxiv.org/abs/2508.17577','https://arxiv.org/abs/2510.11388','https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/kalman_filter.html'],
        derivation='S=sum f_i^2. s_new=S/S0new; k_new=k_old*S0new/S0old. Fixed initial reference implements equivalent coordinates without transporting bounds or queue. Covariance transport DPD^T is standard. Not a novel estimator claim.'))
    print('Stage51 frozen; algebra and actual RLS/rollout invariance passed',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'] and base.inputs()==p['input_hashes'];return p
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.name=='targets.json' or p.is_relative_to((BASE/'raw_only_project/evaluation_data').resolve()):raise RuntimeError('Truth forbidden in forecast replay '+str(p))
def infer():
    sys.addaudithook(guard);verify()
    for case,cfg in CASES.items():
        position_rows=[r for r in rows(base.POSITIONS/case/'observations.jsonl') if cfg['start']<r['t']<=cfg['start']+cfg['duration']]
        forecast={round(r['t'],8):r['frontends']['body_first'] for r in rows(base.POSITIONS/case/'forecasts.jsonl')}
        control={round(r['t'],8):r['frontends']['associated_slotfix'] for r in rows(base.OUT/case/'forecasts.jsonl')}
        if cfg['geometry']:
            geometry={round(r['t'],8):r['observation'] for r in rows(BASE/cfg['geometry'])};sensor={round(r['t'],8):r for r in rows(base.SENSING/case/'sensing.jsonl')}
        models={m:Managed(m) for m in METHODS};selectors={m:Selector() for m in METHODS};folder=OUT/case;folder.mkdir(exist_ok=False)
        stats={m:dict(epoch_changes=0,frequency_range_rejections=0,physics_active=0,actual_updates={n:0 for n in prior.PHYSICS}) for m in METHODS}
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for row in position_rows:
                t=row['t'];key=round(t,8);position=row['methods']['body_first']['observation'];g=geometry[key] if cfg['geometry'] else dict(detected=False);sr=sensor[key] if cfg['geometry'] else None
                obs=observation(position,g);sensed=copy.deepcopy(sr['methods']['replace_stale']) if sr else empty_sensing();sensed,audit=associate(sensed,position);epoch=sr['slot_audit']['calibration_epoch'] if sr else 0
                predictions={};diagnostics={}
                for method in METHODS:
                    q,d=models[method].step(t,obs,sensed,epoch);q=selectors[method].step(q,obs);q['association_diagnostic']=audit
                    assert q['valid']==forecast[key]['valid']
                    for name in prior.robust.MODELS:assert q['coefficients'].get(name)==forecast[key]['coefficients'].get(name),(case,t,method,name)
                    if method=='reset_all':assert q==control[key],(case,t,'original48 control differs')
                    for name,count in d['actual_update_increments'].items():stats[method]['actual_updates'][name]+=count
                    stats[method]['epoch_changes']+=d['epoch_changed'];stats[method]['frequency_range_rejections']+=d['reference']['range_rejected'];stats[method]['physics_active']+=q['feature_diagnostic']['active']
                    predictions[method]=q;diagnostics[method]=d
                f.write(json.dumps(dict(t=t,frontends=predictions,lifecycle=diagnostics),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(position_rows),all_reset_control_exact_to48=True,six_history_exact_to45=True,methods=stats,labels_read=False));print(case,json.dumps(stats),flush=True)
    verify()
def score():
    verify();spec=importlib.util.spec_from_file_location('score51',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=METHODS;m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');m.score()
def report():
    verify();a=load(OUT/'analysis.json');learning=[]
    for case in CASES:
        for method,d in load(OUT/case/'completion.json')['methods'].items():learning.append(dict(case=case,method=method,epochs=d['epoch_changes'],active=d['physics_active'],axis_updates=d['actual_updates']['adaptive_axis_only'],frequency_updates=d['actual_updates']['adaptive_axis_freq'],range_rejections=d['frequency_range_rejections']))
    score_rows=[r for r in a['table'] if r['nominal_h']==.4 and r['case']!='negative5' and r['model'] in ['cv','trusted_acc','adaptive_axis_only','adaptive_axis_freq','adaptive_history_mix','adaptive_all_mix']]
    def table(rr,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],5) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rr)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>51：保留物理学习与频率参考</title><style>body{max-width:1380px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}table{border-collapse:collapse;font-size:13px}th,td{padding:6px;border:1px solid #ccd8df}</style><h1>51：转子换槽时，保留轴向与固定参考物理状态</h1><p>使用48完全相同的700时刻当前位置/桨叶传感器/整机框关联。原reset_all的完整15专家输出逐帧精确回归48，六历史模型各路与45精确。只比较重置生命周期：保留轴向独立模型，或把所有频率输入转换到最初固定S0坐标并保留模型。旧共享频率门槛、四可见学习门槛与0.6秒预热保持，未靠放宽质量产生“学习”。</p><p>S=Σf²。新局部参考改变不等于物理推力突变。参数变换k_new=k_old×S0new/S0old、协方差DPDᵀ、历史s反向缩放可保持预测。本轮采用等价固定参考坐标s=S/S0initial，参数/协方差/队列/约束/激励参考仍在原数值坐标内。实际旧RLS、遗忘、边界、队列及rollout不变性控制已运行；500ms整机失锁仍全部重置，超原0.2–5范围弃权。不能跨未知物体永久保留，也不能消除错桨/谐波错误。</p>'''+table(learning,['case','method','epochs','active','axis_updates','frequency_updates','range_rejections'])+'''<h2>同一目标的0.4秒未来误差</h2>'''+table(score_rows,['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','joint_success'])+'''<p>所有预测保存后才读稀疏人工未来框，均为有效像素动力学，没有质量/风速/推力系数真值。还需与纯历史收缩对照，学习次数增加不代表预测准确率提高。完整分析包含0.1/0.2/0.4秒、共同目标和空场。该轮为缓存因果回放，不是新RAW在线系统。</p><p>相关原始研究：<a href="https://arxiv.org/abs/2604.18289">桨叶相对状态估计</a>、<a href="https://arxiv.org/abs/2508.17577">在线RLS四旋翼辨识</a>、<a href="https://arxiv.org/abs/2510.11388">运动残差估参</a>。固定参考和协方差坐标变换是标准处理，不称新颖算法。</p><p><a href="analysis.json">全部未来评分</a> · <a href="protocol.json">冻结协议/代数控制</a></p></html>'''
    (OUT/'第五十一阶段保留物理学习状态报告.html').write_text(page,encoding='utf-8');save(OUT/'learning_summary.json',dict(table=learning));print('Stage51 report complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score','report']);globals()[p.parse_args().action]()
