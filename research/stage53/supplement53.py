"""Verify real RAW scores and render saved outcomes; no inference changes."""
from pathlib import Path
import argparse,datetime,hashlib,html,json
import cv2,numpy as np
import study53 as current
OUT=current.OUT;ROOT=current.ROOT
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    current.verify();p=OUT/'delivery_supplement.json';assert not p.exists()
    current.save(p,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),
        design='Independent exact score-table regression to51/52 after RAW inference. Render runtime/future errors and video frames at fixed25/50/75 percent times. Not selected by errors or visual success; no model/threshold edits.'))
def verify():
    p=current.load(OUT/'delivery_supplement.json');assert p['helper_sha256']==sha(Path(__file__));current.verify()
def render():
    verify();a=current.load(OUT/'analysis.json');old51=current.load(current.previous.REFERENCE_OUT/'analysis.json');old52=current.load(current.previous.OUT/'analysis.json')
    def key(r):return r['case'],r['frontend'],r['model'],r['nominal_h']
    refs={key(r):r for r in old51['table'] if r['frontend']=='reset_all'};refs.update({key(r):r for r in old52['table']})
    for r in a['table']:assert r==refs[key(r)],key(r)
    current.save(OUT/'score_regression.json',dict(all_future_table_rows_exact_to51_52=len(a['table']),all_horizons=[.1,.2,.4],includes_coverage_joint_success_absence=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=['clear6','side6','train3','negative5'];stats=[current.load(OUT/c/'completion.json') for c in cases]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(4)
    bars=axes[0].bar(x,[r['processing_median_ms'] for r in stats],color=['#277da1','#e76f51','#6a994e','#9c89b8']);axes[0].bar_label(bars,fmt='%.1f',padding=3)
    axes[0].axhline(20,color='#333333',ls='--',label='20 ms budget');axes[0].set_xticks(x,['Clear','Side','Recording 3','Empty']);axes[0].set_ylabel('Processing median (ms)');axes[0].set_title('Actual joint RAW runtime (3 controls)');axes[0].legend(fontsize=8)
    x=np.arange(3)
    for j,(model,label,color) in enumerate([('trusted_acc','History shrinkage','#6a994e'),('adaptive_axis_freq','Axis + frequency','#9c89b8')]):
        values=[next(r['mean_fde_px'] for r in a['table'] if r['case']==c and r['frontend']=='delayed_feedback' and r['model']==model and r['nominal_h']==.4) for c in cases[:3]]
        axes[1].bar(x+(j-.5)*.32,values,.32,label=label,color=color)
    axes[1].set_xticks(x,['Clear (2/3)','Side (5/6)','Recording 3 (11/12)']);axes[1].set_ylabel('Mean 0.4 s endpoint error (px)');axes[1].set_title('Same accuracy as the cached52 control');axes[1].legend(fontsize=8)
    fig.suptitle('RAW-only integration verified; still no stable physical advantage or real time',fontsize=11)
    fig.savefig(OUT/'joint_runtime_accuracy.png',dpi=160);plt.close(fig)
    # Fixed time fractions, including failures, rather than selecting successes.
    montage=np.full((3*390,3*640,3),245,np.uint8);examples=[]
    for row,c in enumerate(cases[:3]):
        cap=cv2.VideoCapture(str(OUT/c/'forecasts.mp4'));count=stats[row]['frames'];cfg=current.CASES[c]
        for col,fraction in enumerate([.25,.5,.75]):
            index=min(count-1,int(count*fraction));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok
            t=cfg['start']+(index+1)*.02;name=f'{c}_{col}.png';cv2.imencode('.png',frame)[1].tofile(str(OUT/name));preview=cv2.resize(frame,(640,360))
            y=row*390;x=col*640;montage[y+30:y+390,x:x+640]=preview
            cv2.putText(montage,f'{c} | {t:.2f}s | fixed {fraction:.0%}',(x+10,y+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(35,55,65),1)
            examples.append(dict(case=c,t=t,frame_index=index,time_fraction=fraction,image=name))
        cap.release()
    cv2.imencode('.png',montage)[1].tofile(str(OUT/'joint_prediction_examples.png'));current.save(OUT/'fixed_examples.json',dict(examples=examples,selection='Fixed25/50/75percent times. No future truth or error selection.'))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>53联合RAW预测结果总览</title><style>body{max-width:1300px;margin:30px auto;font:17px/1.8 Arial;color:#233e4c}img{width:100%}</style><h1>整机定位、桨叶频率与物理学习已接入同一RAW入口</h1><p>14秒正式RAW共700整机/1400桨叶时刻，清晰/侧视/3号/空场；另外短试跑及独立4.5秒前缀。实际运行不读缓存和标签，缓存仅在结束后的独立审计使用。所有540行未来评分与旧51/52精确一致：整合没有新增精度收益，保留了已验证修复。</p><img src="joint_runtime_accuracy.png"><p>单步处理中位清晰225.5ms、侧视182.5ms、3号135.7ms、空场85.4ms。仍超20ms预算；含三路完整物理对照，两个独立RAW解码器用于隔离几何数值状态。不能与仅位置入口直接计算加速倍数。</p><p>下一步先评估只运行最终完整物理分支的部署开销，再考虑共享一次解码的原始事件以减少重复IO，同时保留10/20ms两路隔离与回归。随后将已有整机位置用于桨盘区域检索/共同机架关联；缺四、误桨及姿态质量仍是准确率瓶颈。</p><h2>固定时刻画面，包含无检测/失效时刻</h2><p>各段25%/50%/75%位置，不按预测好坏挑图。橙框：当前整机；绿：历史收缩；紫：轴向频率物理。没有未来真值叠加。</p><img src="joint_prediction_examples.png"><p><a href="第五十三阶段真实RAW联合预测报告.html">完整报告、命令与视频</a> · <a href="raw_prefix_checks.json">独立真实RAW前缀</a> · <a href="score_regression.json">全部评分回归</a></p></html>'''
    (OUT/'联合RAW预测结果总览.html').write_text(page,encoding='utf-8')
    print('540 score rows exact; figures and fixed-time video examples saved',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','render']);globals()[p.parse_args().action]()
