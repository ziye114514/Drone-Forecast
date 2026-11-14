"""Independent cached-input prefix and presentation of already saved results."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json
import study52 as current
OUT=current.OUT;ROOT=current.ROOT
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    current.verify();p=OUT/'prefix_supplement.json';assert not p.exists()
    current.save(p,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),
        case='side6',duration_s=4.5,expected_records=225,design='Fresh model instances, current52 reused inference and51 exact control. Truncate observed stream to112.5s; compare complete forecasts/lifecycle to full-run prefix. Cached current inputs, not a fresh RAW decode. No source change or result tuning.'))
def verify():
    p=current.load(OUT/'prefix_supplement.json');assert p['helper_sha256']==sha(Path(__file__));current.verify();return p
def prefix():
    p=verify();ns=dict(current.previous.__dict__);folder=OUT/'prefix45';folder.mkdir(exist_ok=False)
    cfg=dict(current.CASES[p['case']],duration=p['duration_s'])
    def factory(name):return current.Managed('stable_reference') if name=='stable_reference' else current.Managed52()
    ns.update(OUT=folder,CASES={p['case']:cfg},METHODS=current.METHODS,Managed=factory,REFERENCE_OUT=current.REFERENCE_OUT,verify=current.verify)
    exec(compile(current.infer_source(),'<stage52 independent prefix>','exec'),ns);ns['infer']()
    def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    short=rows(folder/p['case']/'forecasts.jsonl');full=rows(OUT/p['case']/'forecasts.jsonl')
    assert len(short)==p['expected_records'] and short==full[:len(short)]
    current.save(OUT/'prefix_checks.json',dict(independent_cached_current_prefix_records=len(short),complete_forecasts_lifecycle_exact=True,features_and_model_parameters_exact=True,labels_read=False))
    print('225 complete prefix records exact; fresh model instances',flush=True)
def plot():
    verify()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    a=current.load(OUT/'analysis.json');learning=current.load(OUT/'learning_summary.json')['table'];cases=['clear6','side6','train3'];labels=['Clear (2/3)','Side (5/6)','Recording 3 (11/12)'];x=np.arange(3)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for j,(method,label,color) in enumerate([('stable_reference','Stable reference','#277da1'),('delayed_feedback','Delayed feedback','#e76f51')]):
        values=[next(r['frequency_updates'] for r in learning if r['case']==c and r['method']==method) for c in cases]
        bars=axes[0].bar(x+(j-.5)*.3,values,.3,label=label,color=color);axes[0].bar_label(bars,padding=3)
    axes[0].set_xticks(x,['Clear','Side','Recording 3']);axes[0].set_ylabel('Actual RLS updates');axes[0].set_title('Feedback scheduling recovers 10 updates');axes[0].legend(fontsize=8)
    for j,(method,model,label,color) in enumerate([('stable_reference','adaptive_axis_freq','Physical: before','#277da1'),('delayed_feedback','adaptive_axis_freq','Physical: delayed feedback','#e76f51'),('delayed_feedback','trusted_acc','History shrinkage','#6a994e')]):
        values=[next(r['mean_fde_px'] for r in a['table'] if r['case']==c and r['frontend']==method and r['model']==model and r['nominal_h']==.4) for c in cases]
        axes[1].bar(x+(j-1)*.25,values,.25,label=label,color=color)
    axes[1].set_xticks(x,labels);axes[1].set_ylabel('Mean endpoint error (px)');axes[1].set_title('0.4 s: same target coverage');axes[1].legend(fontsize=8)
    rr=[json.loads(line) for line in (OUT/'side6/forecasts.jsonl').read_text(encoding='utf-8').splitlines()]
    for method,label,color in [('stable_reference','Before','#277da1'),('delayed_feedback','Delayed feedback','#e76f51')]:
        axes[2].step([r['t'] for r in rr],[r['frontends'][method]['feature_diagnostic']['models']['adaptive_axis_freq']['updates'] for r in rr],where='post',label=label,color=color)
    axes[2].set_xlabel('Recording time (s)');axes[2].set_ylabel('Cumulative RLS updates');axes[2].set_title('Side sequence');axes[2].legend(fontsize=8)
    fig.suptitle('Cached self-recordings: more learning, still no stable physical advantage',fontsize=12)
    fig.savefig(OUT/'feedback_comparison.png',dpi=160);plt.close(fig)
    # Post-hoc50 reason counts only; never used to change frozen thresholds.
    folder=ROOT.parent/'output/stage50_rigid_slot_association_20261020';summary=[]
    for c in cases:
        rr=[json.loads(x)['methods']['layout']['audit']['layout'] for x in (folder/c/'association.jsonl').read_text(encoding='utf-8').splitlines()]
        summary.append(dict(case=c,initialization_reasons=dict(collections.Counter(r['initialization_reason'] for r in rr)),maximum_initialization_streak=max(r['initializing_streak'] for r in rr)))
    current.save(folder/'initialization_diagnostic.json',dict(table=summary,posthoc_diagnosis_only=True,helper=str(Path(__file__)),helper_sha256=sha(Path(__file__))))
    current.save(OUT/'supplement_checks.json',dict(helper_sha256=sha(Path(__file__)),figure='feedback_comparison.png',prefix_checks=current.load(OUT/'prefix_checks.json')))
    (OUT/'补充图与前缀验证.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>52结果图与前缀验证</title><style>body{max-width:1300px;margin:30px auto;font:17px/1.8 Arial;color:#233e4c}img{width:100%}</style><h1>学习恢复，物理预测尚未稳定优于历史</h1><p>侧视实际更新0→10；同5/6目标的0.4秒物理端点误差123.76→122.46px，纯历史收缩81.91px。清晰段与3号没有改变，不能推广为普遍精度提升。独立4.5秒缓存当前输入前缀225时刻，完整预测和生命周期与全程前缀精确一致；不是新RAW解码。</p><img src="feedback_comparison.png"><p><a href="第五十二阶段历史桨叶与位置反馈报告.html">完整52报告</a> · <a href="prefix_checks.json">前缀证据</a> · <a href="analysis.json">全部评分</a></p></html>''',encoding='utf-8')
    print('Figure and post-hoc50 diagnosis saved',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prefix','plot']);globals()[p.parse_args().action]()
