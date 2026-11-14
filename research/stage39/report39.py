"""Report completed filtering comparison, retaining the numerical isolation failure."""
import html,json
import numpy as np
import study39 as s

def main():
    s.verify();a=s.load(s.OUT/'analysis.json');current=[];stats=[]
    for case,cfg in s.CASES.items():
        completion=s.load(s.OUT/case/'completion.json');stats.append(dict(case=case,**completion))
        if case=='negative5':continue
        labels=[r for r in s.load(s.ROOT.parent/'output/raw_only_project/evaluation_data'/('train3.json' if case=='train3' else 'train6.json'))['labels'] if cfg['start']<r['t']<=cfg['start']+cfg['duration']]
        rr=list(map(json.loads,(s.OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()));tt=np.array([r['t'] for r in rr])
        for method in s.METHODS:
            hits=[];success=0;absent=fp=present=0
            for label in labels:
                i=int(np.searchsorted(tt,label['t']+1e-8,side='right')-1);o=rr[i]['methods'][method]['observation'] if i>=0 and label['t']-tt[i]<=.025 else None
                if label['state']=='absent':absent+=1;fp+=bool(o and o['detected']);continue
                present+=1
                if o and o['detected']:
                    b=np.array(label['bbox']);error=float(np.linalg.norm(np.array(o['center'])-(b[:2]+b[2:])/2));hits.append(error);success+=error<=.25*np.linalg.norm(b[2:]-b[:2])
            current.append(dict(case=case,method=method,present=present,detected=len(hits),center_success=int(success),mean_error_px=float(np.mean(hits)) if hits else None,absent=absent,false_positives=fp))
    s.save(s.OUT/'current_localization.json',current)
    def table(rows,columns):
        return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>' for c in columns)+'</tr>' for r in rows)+'</table>'
    chosen=[r for r in a['table'] if r['model'] in ['cv','trusted_acc'] and r['nominal_h']==.4 and r['case']!='negative5']
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第三十九阶段SDK降噪几何对照</title><style>body{font:16px/1.7 Arial;margin:32px auto;max-width:1280px;color:#192a36}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd6de;padding:7px}th{background:#e8f1f6}h1,h2{color:#154461}</style><h1>第三十九阶段：SDK降噪能否改善桨盘几何与预测</h1>
    <p>原生10ms RAW，仅在一支前端前加持续10ms ActivityNoiseFilter。相同几何、六个历史预测器，先保存预测再读稀疏框。四段共3099时刻；不重算转速/物理，不涉及G。</p>
    <p><b>结论：改善不统一，跨录像漏检仍未解决，不升级默认。</b>清晰发布478→498，侧视796→803，train3仍163。train3保留事件仅35.24%，但没有重新找到后续目标，说明去噪不足以修复整机识别。</p>
    <p>0.4秒同样本：清晰CV21.89→23.50px略差，收缩加速度45.45→15.02改善；侧视CV127.62→130.53略差、收缩98.83→90.40改善，但联合成功7→6。train3 CV208.52→207.52，仍只有3/12可预测且联合成功0。开发窗口小，不能从最佳组合选出普适胜者。</p><h2>当前定位</h2>'''
    doc+=table(current,['case','method','present','detected','center_success','mean_error_px','absent','false_positives'])+'<h2>0.4秒真实未来端点误差</h2>'+table(chosen,['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','max_fde_px','joint_success'])
    doc+='<p>共同目标数量clear7、side12、train3 3，两路覆盖一致；全部六预测器/0.1、0.2、0.4秒与共同目标结果见<a href="analysis.json">完整评分</a>。可见粗框中心不是质心，无姿态/RPM真值、无密集ADE。</p><h2>运行与复现</h2>'
    doc+=table([dict(case=x['case'],frames=x['frames'],published=x['published'],retention=round(x['retention'],4),median_ms=x['processing_median_ms'],p95_ms=x['processing_p95_ms']) for x in stats],['case','frames','published','retention','median_ms','p95_ms'])
    doc+='''<p>原基线三段2299观测及六模型系数全部精确重现。train3配对首次在26.17秒出现一个近线段桨盘的OpenCV拟合差异；单路150时刻无差异。保留<a href="baseline_failure_audit.json">诊断</a>及failed_paired_train3，另冻结<a href="isolation_supplement.json">独立进程补充协议</a>后重跑train3两支，完整600时刻基线精确重现。未放宽容差、未修改冻结算法。clear/side为配对计时、train3/负例为独立进程，不把这些时间当严格硬件加速对比；每支中位仍约20–34ms，排除解码写入，未达100Hz。</p>
    <p>负例800时刻两路当前和预报均0。已用SDK标准方法，<a href="https://docs.prophesee.ai/stable/api/python/cv/bindings.html#metavision_sdk_cv.ActivityNoiseFilterAlgorithm">Prophesee官方API</a>，未提出新的降噪算法。下一步用整机监督应对局部桨盘与整机身份缺失。</p></html>'''
    (s.OUT/'第三十九阶段SDK降噪几何报告.html').write_text(doc,encoding='utf-8')
    (s.OUT/'阶段结论.md').write_text('SDK降噪带来局部预测收益但不统一，跨录像当前漏检不变。基线2299逐项精确，负例800无发布。配对OpenCV退化拟合差异和隔离进程补充保留。未推广默认；详见HTML和analysis.json。',encoding='utf-8')
    print('Stage39 report complete',flush=True)
if __name__=='__main__':main()
