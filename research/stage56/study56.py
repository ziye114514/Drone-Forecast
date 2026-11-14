"""Reuse unchanged50 joint layout association on55 current observations."""
from pathlib import Path
from collections import Counter
import argparse, copy, datetime, hashlib, html, json, os, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage55'),str(ROOT/'research/stage50')]
import study55 as previous
import layout50 as layout
OUT=ROOT.parent/'output/stage56_roi_layout_association_20261021'
CASES=[prefix+c for c in ['clear6','side6','train3'] for prefix in ['off_','']]
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows

def sources():
    result=previous.sources()
    for name in ['research/stage56/study56.py','research/stage50/layout50.py','research/stage4/tracker.py','research/stage33/slots33.py']:
        result[name]=sha(ROOT/name)
    return result

def inputs():
    return {f'{c}/{n}':sha(previous.OUT/c/n) for c in CASES for n in ['observations_forecasts.jsonl','rotor_ticks.jsonl']}

def freeze():
    assert not OUT.exists();previous.verify()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=inputs(),cases=CASES,
        parameters=layout.PARAMS,
        design='Same native10 current geometry from55 off/on, unchanged33 baseline vs unchanged50 LayoutSlots. Body hints copied from actual55 requests and known by20ms packet start. Source mixes current observations and saved forecasts, but preparation selects only observation/t/body_hint_sent; association opens only those current-only extracts.',
        controls='Baseline current geometry/IDs/lifecycle must exactly match actual55. Independently execute first120 native ticks per case; all replayed outputs exactly match. Histories empty: no new frequency or forecast execution and no claim of physical accuracy gain.',
        prior_art=['https://arxiv.org/html/2609.11920v1','https://github.com/daniilidis-group/event_feature_tracking'],
        actual_reuse='Unmodified50 LayoutSlots and4 initialization/fit_affine,33 ReplaceStale and original RotorSensing association arithmetic. No threshold tuning, virtual measurements or new geometry method.',
        limits='Candidate identity and pose are unlabelled; retained IDs are not certified motor identities. Tests are cached causal current-state association, not a fresh RAW deployment benchmark.'))
    print('Stage56 frozen: actual unchanged50 on same55 current observations',flush=True)

def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();previous.verify();return p

def prepare():
    verify()
    for case in CASES:
        folder=OUT/case;folder.mkdir(exist_ok=False)
        parent=rows(previous.OUT/case/'observations_forecasts.jsonl');hints={}
        for r in parent:
            packet_start=round((r['t']-.02)*1e6);hint=r['body_hint_sent']
            if hint is not None:assert round(hint['t']*1e6)<=packet_start
            hints[round((r['t']-.01)*1e6)]=hint;hints[round(r['t']*1e6)]=hint
        with (folder/'current_only.jsonl').open('x',encoding='utf-8') as stream:
            for r in rows(previous.OUT/case/'rotor_ticks.jsonl'):
                item=dict(t=r['t'],observation=r['observation'],body_hint=hints[round(r['t']*1e6)],
                    reference_geometry=layout.geometry_record(r['sensing']),reference_four_visible=r['sensing']['four_visible'],reference_epoch=r['slot_audit']['calibration_epoch'],reference_retired=r['slot_audit']['retired_ids'])
                stream.write(json.dumps(item,allow_nan=False)+'\n')
        assert len(hints)==len(parent)*2
    save(OUT/'current_input_hashes.json',{c:sha(OUT/c/'current_only.jsonl') for c in CASES})
    print('Current-only causal extracts saved; no future labels read',flush=True)

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.name in ['observations_forecasts.jsonl','forecasts.jsonl','analysis.json','targets.json'] or p.is_relative_to(ROOT.parent/'output/raw_only_project/evaluation_data'):
            raise RuntimeError('Association replay cannot open future/truth file '+str(p))

def execute(records):
    sensors={'replace_stale':layout.ReplaceStale(layout.PaperConfig()),'layout':layout.LayoutSlots(layout.PaperConfig())}
    result=[]
    for r in records:
        t=r['t'];item=dict(t=t,methods={})
        for name,sensor in sensors.items():
            sensed=sensor.step(copy.deepcopy(r['observation']),[],t,copy.deepcopy(r['body_hint'])) if name=='layout' else sensor.step(copy.deepcopy(r['observation']),[],t)
            item['methods'][name]=dict(rotors=layout.geometry_record(sensed),four_visible=sensed['four_visible'],audit=copy.deepcopy(sensor.audit),new_ids=sensor.next_id)
        result.append(item)
    return result

def run():
    verify();h=load(OUT/'current_input_hashes.json')
    for c,v in h.items():assert sha(OUT/c/'current_only.jsonl')==v
    sys.addaudithook(guard);table=[];gate=[];total=0;prefix_ticks=0
    for case in CASES:
        rr=rows(OUT/case/'current_only.jsonl');result=execute(rr);prefix=execute(rr[:120]);assert result[:len(prefix)]==prefix
        total+=len(result);prefix_ticks+=len(prefix)
        for actual,reference in zip(result,rr):
            baseline=actual['methods']['replace_stale']
            assert baseline['rotors']==reference['reference_geometry'] and baseline['four_visible']==reference['reference_four_visible']
            assert baseline['audit']['calibration_epoch']==reference['reference_epoch'] and baseline['audit']['retired_ids']==reference['reference_retired']
        changed=sum(r['methods']['replace_stale']['rotors']!=r['methods']['layout']['rotors'] for r in result)
        with (OUT/case/'association.jsonl').open('x',encoding='utf-8') as stream:
            for r in result:stream.write(json.dumps(r,allow_nan=False)+'\n')
        tester=layout.LayoutSlots(layout.PaperConfig());streak=maximum=eligible=0;pending=None;reasons=Counter()
        for r in rr:
            initial,reason=layout.skeleton.initialization(r['observation'].get('rotors',[]))
            if initial is not None and not tester.inside(r['observation']['rotors'],r['body_hint'],r['t']):initial=None;reason='body_hint_missing_or_outside'
            reasons[reason]+=1
            if initial is None:streak=0;pending=None;continue
            eligible+=1;center=initial[0].mean(0);streak=streak+1 if pending is not None and np.linalg.norm(center-pending)<50 else 1;pending=center;maximum=max(maximum,streak)
        for name in ['replace_stale','layout']:
            mm=[r['methods'][name] for r in result]
            table.append(dict(case=case,method=name,ticks=len(result),four_visible=sum(r['four_visible'] for r in mm),
                retired=sum(len(r['audit']['retired_ids']) for r in mm),new_ids=mm[-1]['new_ids'],epochs=mm[-1]['audit']['calibration_epoch'],
                joint_association=sum(r['audit'].get('layout',{}).get('joint_association',False) for r in mm),
                initializations=sum(r['audit'].get('layout',{}).get('initialized_now',False) for r in mm),
                maximum_track_initializing_streak=max(r['audit'].get('layout',{}).get('initializing_streak',0) for r in mm),changed_ticks=changed if name=='layout' else 0))
        gate.append(dict(case=case,eligible_raw_current_ticks=eligible,maximum_raw_current_streak=maximum,required_streak=layout.skeleton.PARAMS['init_frames'],reasons=dict(reasons)))
        save(OUT/case/'completion.json',dict(baseline_geometry_ids_lifecycle_exact55=True,prefix_exact_ticks=len(prefix),ticks=len(result),frequency_or_forecast_executed=False))
    assert total==2600 and prefix_ticks==720
    save(OUT/'results.json',dict(table=table,initialization=gate,total_cached_ticks=total,prefix_ticks=prefix_ticks,no_labels_read=True,no_frequency_or_forecast_executed=True))
    def tab(rr,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[x]))+'</td>' for x in cols)+'</tr>' for r in rr)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>56：ROI之后共同机架关联复用</title><style>body{max-width:1400px;margin:28px auto;font:16px/1.8 Arial;color:#233e4c}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd8df}</style><h1>56：不改原门槛，复用共同机架关联器</h1><p>在55两种几何的同一批当前观测上，直接调用原50 LayoutSlots、4初始化和33 ReplaceStale。整机提示来自实际55发送请求，已知时间不晚于20ms包开始；不使用未来NN框。当前关联过程只读取剥离后的current_only文件，禁止打开预测/真值文件。</p>'''+tab(table,list(table[0]))+'''<h2>直接当前四候选的初始化条件</h2>'''+tab(gate,['case','eligible_raw_current_ticks','maximum_raw_current_streak','required_streak','reasons'])+'''<p>原五帧、尺寸、闭合、扁平度、短时支持门槛不变。基线2600当前时刻的几何、ID、生命周期逐项精确55，每case独立前120时刻共720前缀精确。没有执行新频率或未来预测，不由ID保留推断机械转子身份真值。是否接完整RAW必须先看输出是否实际改变，不能为相同结果重复解码。</p><p>已有基础：<a href="https://arxiv.org/html/2609.11920v1">EVPeriscope</a>持续桨心/已知机架，<a href="https://github.com/daniilidis-group/event_feature_tracking">事件特征跟踪作者代码</a>提供模型与关联思路；本轮实际执行的是已有本地4/33/50，不是上述完整系统。</p><p><a href="results.json">完整统计</a> · <a href="protocol.json">协议</a> · <a href="../stage55_body_guided_rotors_20261021/桨盘提取对照结果与固定时刻画面.html">55完整RAW预测结果</a></p></html>'''
    (OUT/'第五十六阶段ROI机架关联复用报告.html').write_text(page,encoding='utf-8')
    save(OUT/'completion.json',dict(source_hashes_unchanged=load(OUT/'protocol.json')['source_hashes']==sources(),total_cached_ticks=total,prefix_ticks=prefix_ticks,baseline_exact55=True,changed_ticks_total=sum(r['changed_ticks'] for r in table)))
    for r in table:print(r,flush=True)
    for r in gate:print(r,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','run']);globals()[p.parse_args().action]()
