from pathlib import Path
import argparse,datetime,hashlib,html,json,os,sys
import numpy as np
from layout50 import *
from checks50 import run as checks
BASE=ROOT.parent/'output';OUT=BASE/'stage50_rigid_slot_association_20261020';BODY=BASE/'stage45_cuda_graph_20261019/sequence';SENSING=BASE/'stage34_slot_frequency_forecast_20261018'
CASES={'clear6':dict(start=22.,duration=2.,observation='stage12_clear_22_27_20261017/observations.jsonl'),
    'side6':dict(start=108.,duration=5.,observation='stage12_side_108_120_20261017/observations.jsonl'),
    'train3':dict(start=25.,duration=6.,observation='stage31_cross_recording_physics_20261018/train3/observations_forecasts.jsonl')}
METHODS=['replace_stale','layout']
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sources():
    paths=[Path(__file__).with_name(n) for n in ['layout50.py','checks50.py','study50.py']]+[ROOT/'research/stage4/tracker.py',ROOT/'research/stage33/slots33.py',ROOT/'src/event_drone_forecast/rotor_sensing.py',ROOT/'src/event_drone_forecast/paper_config.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def inputs():return {p.relative_to(BASE).as_posix():sha(p) for c,cfg in CASES.items() for p in [BASE/cfg['observation'],BODY/c/'observations.jsonl',SENSING/c/'sensing.jsonl']}
def freeze():
    assert not OUT.exists(),'Preserve previous results.'
    check=checks();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=inputs(),cases=CASES,parameters=PARAMS,module_checks=check,
        design='1300 cached native10 current geometry ticks, unchanged33 baseline geometry/IDs exact34. Past-or-current45 body_only boxes at20ms, held<=160ms. Joint association only with four genuine current candidates. EMPTY event histories: association only, no new RPM or forecast gain claim. Independent prefix replay.',
        prior_art=['https://arxiv.org/html/2609.11920v1','https://github.com/daniilidis-group/event_feature_tracking','https://github.com/tobycollins/IPPE'],
        executed_reuse='Actual local4 fit_affine/initialization/X +33 ReplaceStale/SciPy Hungarian; external full systems not executed. No GT/manual seed or G.'))
    print('Stage50 frozen with module checks',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();return p
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.name in ['targets.json','forecasts.jsonl'] or p.is_relative_to((BASE/'raw_only_project/evaluation_data').resolve()):raise RuntimeError('Truth/future forbidden '+str(p))
def current_body(body_rows,t):
    valid=[r for r in body_rows if r['t']<=t+1e-8]
    return valid[-1]['methods']['body_only']['observation'] if valid else None
def execute(rr,bb):
    sensors={'replace_stale':ReplaceStale(PaperConfig()),'layout':LayoutSlots(PaperConfig())};result=[]
    for r in rr:
        t=r['t'];body=current_body(bb,t);record=dict(t=t,methods={})
        for name,s in sensors.items():
            q=s.step(r['observation'],[],t,body) if name=='layout' else s.step(r['observation'],[],t)
            record['methods'][name]=dict(rotors=geometry_record(q),four_visible=q['four_visible'],audit=s.audit,new_ids=s.next_id)
        result.append(record)
    return result
def run():
    sys.addaudithook(guard);verify();table=[]
    for c,cfg in CASES.items():
        rr=[r for r in rows(BASE/cfg['observation']) if cfg['start']<r['t']<=cfg['start']+cfg['duration']];bb=rows(BODY/c/'observations.jsonl');reference={round(r['t'],8):r for r in rows(SENSING/c/'sensing.jsonl')}
        result=execute(rr,bb);assert len(result)==round(cfg['duration']*100)
        for r in result:assert r['methods']['replace_stale']['rotors']==geometry_record(reference[round(r['t'],8)]['methods']['replace_stale']),(c,r['t'],'baseline differs')
        prefix=execute(rr[:120],[r for r in bb if r['t']<=rr[min(119,len(rr)-1)]['t']+1e-8]);assert prefix==result[:len(prefix)]
        folder=OUT/c;folder.mkdir(exist_ok=False)
        with (folder/'association.jsonl').open('x',encoding='utf-8') as f:
            for r in result:f.write(json.dumps(r,allow_nan=False)+'\n')
        changed=sum(r['methods']['replace_stale']['rotors']!=r['methods']['layout']['rotors'] for r in result)
        for name in METHODS:
            mm=[r['methods'][name] for r in result];table.append(dict(case=c,method=name,frames=len(result),four_visible=sum(m['four_visible'] for m in mm),
                retired=sum(len(m['audit']['retired_ids']) for m in mm),calibration_resets=mm[-1]['audit']['calibration_epoch'],new_ids=mm[-1]['new_ids'],
                layout_joint_current=sum(m['audit'].get('layout',{}).get('joint_association',False) for m in mm),
                layout_initializations=sum(m['audit'].get('layout',{}).get('initialized_now',False) for m in mm),geometry_or_id_changed_ticks=changed if name=='layout' else 0))
        save(folder/'completion.json',dict(baseline_geometry_exact_to34=len(result),prefix_current_records_exact=len(prefix),frequency_estimation_executed=False,sources_unchanged=sources()==load(OUT/'protocol.json')['source_hashes']))
    save(OUT/'analysis.json',dict(table=table,module_checks=load(OUT/'protocol.json')['module_checks']))
    columns=list(table[0]);tab='<table><tr>'+''.join('<th>'+x+'</th>' for x in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[x]))+'</td>' for x in columns)+'</tr>' for r in table)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>50：共同机架辅助转子关联</title><style>body{max-width:1350px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}td,th{border:1px solid #ccd8df;padding:6px}table{border-collapse:collapse;font-size:13px}</style><h1>50：复用已有仿射机架，辅助转子关联</h1><p>将第4阶段fit_affine/初始化与第33阶段ReplaceStale实际复用，在同样1300个真实缓存当前几何上比较。只在四个当前观测、机架形状/运动/当前整机框约束通过时联合关联；没有补出桨心，没有伪造频率。NN框仅提供共同位移，不是中心真值。不能由保留ID数推断真实机械转子身份正确。</p>'''+tab+'''<p>基线逐帧几何/ID与34完全一致，独立前120时刻回放一致。控制仿真有已知点身份和独立中心测量，验证共同位移跳变后保留四身份；无整机证据时必须与原33完全一致，空输入后过期。</p><p>当前事件history为空，故本阶段只验证关联，未执行新频率/未来预测。是否值得接RAW频率需看变化和失败；不升级默认。四桨和透视/共线仍限制机架可用性。</p><p>相关原始研究<a href="https://arxiv.org/html/2609.11920v1">EVPeriscope</a>用持续桨心及已知不对称机架PnP/EKF，仍要求四心可见；其项目页本轮访问失败，未确认可直接复用的作者代码。<a href="https://github.com/daniilidis-group/event_feature_tracking">事件特征概率关联作者代码</a>提供仿射模板/关联基础，本轮没有执行MATLAB/IMU系统。实际执行的是本项目已有4/33模块及SciPy。</p><p><a href="analysis.json">统计</a> · <a href="protocol.json">协议与控制检查</a></p></html>'''
    (OUT/'第五十阶段机架辅助桨心关联.html').write_text(page,encoding='utf-8');verify();print(json.dumps(table,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);globals()[p.parse_args().action]()
