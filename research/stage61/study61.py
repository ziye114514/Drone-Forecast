"""Freeze and validate the pre-ellipse line route against the measured RNG failure."""
from pathlib import Path
import argparse,datetime,hashlib,html,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage60')]
import study60 as previous
import isolated60
import preline61 as preline
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage61_preellipse_lines_20261021'
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.previous.rows
PARAMS=dict(seed_modes=['normal','consume_rng'],cases=dict(clear6=dict(start=22.,duration=.8,raw=r'E:\dataset\train\6\media\event.raw'),
    side6=dict(start=112.,duration=.8,raw=r'E:\dataset\train\6\media\event.raw'),train3=dict(start=25.,duration=1.2,raw=r'E:\dataset\train\3\aligned\event.raw'),
    negative5=dict(start=0.,duration=.2,raw=r'E:\dataset\train\5\media\event.raw')),
    design='Different OpenCV RNG-stream histories in independent processes, identical RAW and cached past57 body hints. Stable and stable-paired independent states. Test current observations, preline routing and candidate audits exactly across RNG interventions; no forecast/label scoring. Same full61 future protocol will follow only after stability passes.')

def sources():
    r=isolated60.sources()
    for p in [Path(__file__),Path(__file__).with_name('preline61.py')]:r[p.relative_to(ROOT).as_posix()]=sha(p)
    return r
def freeze():
    isolated60.verify();assert not OUT.exists();cause=load(previous.OUT/'rng_cause60.json')
    assert cause['fixed_seed_all_equal'] and cause['consumption_intervention_changed_fit'] and cause['sequential_fit_changes']
    checks=preline.checks();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,geometry_parameters=preline.PARAMS,
       checks=checks,derived_candidates_sha256=hashlib.sha256(preline.DERIVED_SOURCE.encode()).hexdigest(),cause_sha256=sha(previous.OUT/'rng_cause60.json'),
       scope='Thin contours measured as lines before any ellipse fit. Existing broad ellipse/partial/grouping/frequency code unchanged. Reject unmeasurable line, not random thin ellipse. No production promotion, rotor identity or new future forecast claim.'))
    (OUT/'derived_candidates61.py').write_text(preline.DERIVED_SOURCE,encoding='utf-8');print('61 pre-ellipse line route frozen',flush=True)
def verify():
    isolated60.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['derived_candidates_sha256']==hashlib.sha256(preline.DERIVED_SOURCE.encode()).hexdigest()
def run(mode,case):
    verify();path=OUT/f'{mode}_{case}.json';assert not path.exists();cfg=PARAMS['cases'][case];reference=isolated60.REF/case;hints={}
    for p in rows(reference/'observations_forecasts.jsonl'):
        hint=p['body_hint_sent']
        if hint is not None:assert round(hint['t']*1e6)<=round((p['t']-.02)*1e6)
        for t in [p['t']-.01,p['t']]:hints[round(t*1e6)]=hint
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();stable=preline.StableFrontend(w,h,'neighbor10');paired=preline.StablePairedFrontend(w,h,'neighbor10','paired');result=[];cv2.setRNGSeed(123)
    for t,e in raw_chunks(reader,cfg['start'],cfg['duration'],10000):
        if mode=='consume_rng':
            junk=np.zeros((32,32),np.float32);cv2.randu(junk,0.,1.)
        for f in [stable,paired]:f.set_hint(e,hints[round(t*1e6)],t,'body')
        a,_=stable.detect(e,t);b,_=paired.detect(e,t)
        result.append(dict(t=t,stable=a,paired=b,stable_preline=stable.preline_audit,paired_preline=paired.preline_audit,line_audit=paired.line_audit))
    assert len(result)==round(cfg['duration']*100);save(path,dict(records=result));print(case+' '+mode+' complete '+str(len(result)),flush=True)
def report():
    verify();table=[]
    for case in PARAMS['cases']:
        a=load(OUT/f'normal_{case}.json')['records'];b=load(OUT/f'consume_rng_{case}.json')['records'];assert a==b,(case,'RNG intervention still changes current geometry')
        table.append(dict(case=case,native10=len(a),RNG_intervention_exact=True,
            thin_routed=sum(r['stable_preline']['thin_routed'] for r in a),thin_rejected=sum(r['stable_preline']['thin_rejected'] for r in a),
            line_measured=sum(r['stable_preline']['line_measured'] for r in a),stable_paired_changed_ticks=sum(r['stable']!=r['paired'] for r in a)))
    save(OUT/'results.json',dict(table=table,unique_native10=300,total_RAW_window_reads=600,labels_read=False,new_future_scoring=False))
    cols=list(table[0]);tab='<table><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[c]))+'</td>' for c in cols)+'</tr>' for r in table)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>61 先判线再拟合椭圆</title><style>body{max-width:1400px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccd8df}</style><h1>61：侧视薄轮廓在椭圆拟合之前走线段路径</h1><p>60实际复现了旧路径的重要问题：OpenCV5对同一个443,293,1,12的真实共线点集，连续拟合长度从约9.86到13.23px；固定seed结果一致、预先消耗随机数会改变拟合。旧路径先fitEllipse再贴line标签，随机状态能改变尺度门槛和三/四点分组。该控制直接验证了RNG对这个轮廓的影响；没有给未知原失败补造首个观测证据。</p><p>新适配器只在原候选循环的fitEllipse之前插入薄轮廓分支，调用已有geometry._line，太短或无效就拒绝，不回退病态椭圆。原宽椭圆/残弧/分组/频率代码与原源保持，使用MRO实际复用9/8/57包装；没有在正式推理设随机种子。模块检查禁止真实共线轮廓调用fitEllipse，已知20px长线中心准确，宽椭圆完整前端精确旧路径。</p>'''+tab+'''<p>300个不同当前RAW时刻、两种随机状态历史共600窗口读取，含26.17故障时刻；新普通/带配对分支的所有当前观测及审计逐项相同。只是可重复性和测量模型修复，不是桨心/姿态真值或新的未来预测精度。旧默认仍54，下一步必须以稳定线段基线、稳定线段加配对两路完整频率/15模型和真实RAW前缀评分，才能判断预测收益。</p><p><a href="protocol.json">冻结协议与检查</a> · <a href="results.json">全部当前控制</a> · <a href="../stage60_paired_lines_20261021/rng_cause60.json">真实点集RNG因果控制</a> · <a href="../stage60_paired_lines_20261021/degenerate_contour_probe.json">全部重复/换序拟合</a></p></html>'''
    (OUT/'第六十一阶段先判线再拟合椭圆报告.html').write_text(page,encoding='utf-8');print(table,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','report']);p.add_argument('--mode',choices=PARAMS['seed_modes']);p.add_argument('--case',choices=list(PARAMS['cases']));a=p.parse_args()
    if a.action=='run':run(a.mode,a.case)
    else:globals()[a.action]()
