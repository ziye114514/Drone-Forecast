"""Frozen pairing/scale/temporal diagnostic on all existing RAW line observations."""
from pathlib import Path
import argparse,datetime,html,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/stage59'))
import study59 as previous
import lsd59
import pairs60 as pairs
OUT=ROOT.parent/'output/stage60_paired_lines_20261021'
load=previous.load;save=previous.save;sha=previous.sha;CASES=previous.CASES

def sources():
    r=previous.sources()
    for p in [Path(__file__),Path(__file__).with_name('pairs60.py'),ROOT/'research/stage59/lsd59.py']:r[p.relative_to(ROOT).as_posix()]=sha(p)
    return r

def inputs():return {c+'/'+n:sha(previous.OUT/n) for c in CASES for n in [c+'.json','lsd_'+c+'.json']}

def freeze():
    previous.verify();lsd59.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=inputs(),parameters=pairs.PARAMS,checks=pairs.checks(),
       design='All450 current RAW line records from59, no raw re-decode or future fields/labels. Pair opposite gradient-oriented LSD edges by parallelism/overlap/endpoint agreement/width, then one-to-one across fixed1/1.5 thresholds and3 native-tick causal confirmation with past body-scale gate. Complete endpoints and rotor identity remain unverified; no virtual measured hubs or forecast improvement claim.',
       next_gate='If confirmed candidates exist, inspect full density profiles on actual RAW to validate paired interiors/endpoints and compare same existing whole-body position/full15 forecasts. Do not infer motor identity from count increase alone.',
       prior_art=['https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html','https://github.com/event-driven-robotics/RT-EvLDT','https://github.com/uzh-rpg/line_tracking_with_event_cameras','https://github.com/iago-suarez/ELSED'],
       actual_reuse='Original59 OpenCV LSD outputs, installed scipy linear_sum_assignment and old3-tick confirmation/gap conventions. Simple standard stripe pairing, not a reproduction of PowerLine/RT-EvLDT/ELSED.'))
    print('60 pairing frozen before cached replay',flush=True)

def verify():
    previous.verify();lsd59.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs()

def run():
    verify();assert not (OUT/'results.json').exists();summaries=[]
    for case in CASES:
        base=load(previous.OUT/f'{case}.json')['records'];lsd=load(previous.OUT/f'lsd_{case}.json')['records'];track=pairs.Confirm();records=[]
        for a,b in zip(base,lsd):
            assert a['t']==b['t'];variants={}
            for name in ['base','hard1p5','hard2']:
                lines=[v['endpoints'] for v in b['variants'][name]];variants[name]=pairs.pair_segments(lines)
            stable=pairs.cross_threshold(variants['base'],variants['hard1p5']);confirmed=track.step(a['t'],stable,a['roi'])
            records.append(dict(t=a['t'],roi=a['roi'],pairs=variants,stable=stable,tracks=confirmed))
        assert len(records)==round(CASES[case]['duration']*100)
        save(OUT/f'{case}.json',dict(records=records))
        summaries.append(dict(case=case,ticks=len(records),base_pairs=sum(len(r['pairs']['base']) for r in records),
           frames_with_pairs=sum(bool(r['pairs']['base']) for r in records),stable_pairs=sum(sum(p['cross_threshold_stable'] for p in r['stable']) for r in records),
           confirmed_candidates=sum(sum(p['confirmed_candidate'] for p in r['tracks']) for r in records),
           confirmed_frames=sum(any(p['confirmed_candidate'] for p in r['tracks']) for r in records),
           max_simultaneously_confirmed=max([sum(p['confirmed_candidate'] for p in r['tracks']) for r in records]+[0])))
    save(OUT/'results.json',dict(table=summaries,current_ticks=450,no_new_RAW_reads=True));print(summaries,flush=True)

def report():
    verify();rows=load(OUT/'results.json')['table'];cols=list(rows[0]);table='<table><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[c]))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>60 双边配对与时间确认</title><style>body{max-width:1450px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccd8df}</style><h1>60：LSD双边合中心线，跨阈值与过去时间确认</h1><p>复用59全部450个当前RAW线记录，不挑好帧、不读取未来预报/标签。按标准条带几何配对梯度方向相反的平行边，要求沿线重叠、两端对齐、宽度比例合理，再在固定1/1.5阈值间一对一匹配，连续3个原生时刻与过去整机框尺度确认。不是新事件线检测器或已验证的真实电机身份；机臂条带也能满足几何，必须保留这一反例。</p>'''+table+'''<p>此处confirmed仅为几何候选确认，不是完整端点或真实轴心验证。没有把候选数量当准确率、没有新增未来预测成绩。下一步按冻结门槛在实际RAW密度中检查条带内部、两端支持和时序一致性，再进入相同整机位置/完整15模型的小对照。旧54/57默认保持。</p><p><a href="protocol.json">冻结协议/模块反例</a> · <a href="results.json">完整统计</a> · <a href="../stage59_side_lines_20261021/结论与分支融合方案.html">前一轮真实RAW/固定画面</a> · <a href="https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html">实际使用LSD文档</a> · <a href="https://github.com/uzh-rpg/line_tracking_with_event_cameras">PowerLine作者库（本轮未执行）</a></p></html>'''
    (OUT/'第六十阶段双边中心线与尺度确认报告.html').write_text(page,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','report']);globals()[p.parse_args().action]()
