"""Plot observed legacy instability and preserve the incomplete60 future attempt."""
from pathlib import Path
import datetime,json
import numpy as np
import study61 as current

def run():
    current.verify();result=current.load(current.OUT/'results.json');assert result['unique_native10']==300 and result['total_RAW_window_reads']==600
    probe=current.load(current.previous.OUT/'degenerate_contour_probe.json');cause=current.load(current.previous.OUT/'rng_cause60.json')
    contour=next(r for r in probe['contours'] if r['bbox']==[443,293,1,12]);lengths=[max(r['fit'][1]) for r in contour['fits']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    p=np.array(contour['points']);along=p[:,1]-p[:,1].min()
    axes[0].plot(along,np.zeros(len(along)),'o',color='#277da1');axes[0].set(xlabel='Actual vertical support coordinate (px)',ylabel='Same x coordinate',yticks=[0],title='Real collinear contour: support span11px');axes[0].set_ylim(-1,1)
    axes[1].plot(np.arange(len(lengths))+1,lengths,'o-',ms=3,color='#e76f51');axes[1].axhline(11,color='#277da1',label='Visible support span');axes[1].axhline(12,color='gray',ls='--',label='Existing line minimum');axes[1].set(xlabel='Repeated/cyclic/reversed fit index',ylabel='fitEllipse major length (px)',title='Legacy ellipse length is not a stable line length');axes[1].legend(fontsize=8)
    for key,label,color in [('fixed_seed','Reset seed123','#277da1'),('random_numbers_consumed_before_fit','Same seed + unrelated RNG use','#e76f51'),('sequential_same_array_fits','Sequential same array','#43aa8b')]:
        values=[max(f[1]) for f in cause[key]];axes[2].plot(np.arange(len(values))+1,values,'o-',label=label,color=color)
    axes[2].set(xlabel='Controlled repeat',ylabel='Major length (px)',title='Only RNG-stream intervention changes fit');axes[2].legend(fontsize=7)
    fig.suptitle('Actual installed OpenCV5.0.0; identical real point set. No rotor/pose truth inferred.',fontsize=11)
    fig.savefig(current.OUT/'rng_ellipse_lengths.png',dpi=160);plt.close(fig)
    pagepath=current.OUT/'第六十一阶段先判线再拟合椭圆报告.html';page=pagepath.read_text(encoding='utf-8')
    page=page.replace('<style>','<style>img{width:100%}')
    page=page.replace('<table>','<img src="rng_ellipse_lengths.png"><table>',1)
    page=page.replace('</body>','') if '</body>' in page else page
    page=page.replace('</html>','<p>60完整预测尝试保留为未完成：关闭试跑40/80完整15精确旧57，开启clear100/200和side250/500通过，train3完成300/600推理但26.17当前几何回归失败，未执行negative、真实前缀或未来评分。旧默认54保持，未把60原始差异计为改进。</p></html>')
    pagepath.write_text(page,encoding='utf-8')
    current.save(current.OUT/'delivery61.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),
        new_future_scoring=False,default_promoted=False,legacy_cv_version='5.0.0',observed_major_range_px=[min(lengths),max(lengths)],
        real_support_span_px=11.,all_RNG_controls_exact=True,unique_native10=300,total_RAW_window_reads=600,
        failed60_future_protocol_not_complete=True,stage60_terminal_batches=[89399,31005,18571,1565,26481,1447,57994,37342],stage61_terminal_batch=75737))
    print('61 plot/report delivery complete',flush=True)

if __name__=='__main__':run()
