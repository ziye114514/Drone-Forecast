"""Bounded177--180 report. Historical development clips; no benchmark claims."""
from pathlib import Path
import json,html,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output';OUT=BASE/'rotor_inputs177_180_20261107'

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    score=json.loads((BASE/'isolated177_20261107/integrated/scores.json').read_text('utf-8'))
    phase=json.loads((BASE/'helix_bridge180_20261107/review_C_ring_phases.json').read_text('utf-8'))
    pilot=json.loads((BASE/'helix_bridge180_20261107/native_pilot_C.json').read_text('utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));ax=axes[0];x=np.arange(4);w=.25
    for j,(field,label) in enumerate([('discovery_concentration','前半窗口'),('validation_concentration','后半窗口'),('phase_agreement','相位衔接')]):
        ax.bar(x+(j-1)*w,[r['phase'][field] for r in phase['rotors']],w,label=label)
    ax.axhline(.5,color='#a22',ls='--',lw=1);ax.set_xticks(x,[f"候选{i+1}" for i in x]);ax.set_ylim(0,1);ax.set_title('C帧：仅候选3同时通过当前窗口核验');ax.legend(fontsize=8)
    ax=axes[1]
    for branch in pilot['branches']:
        rs=[r for r in branch['records'] if 'state' in r]
        ax.plot([r['t']-pilot['seed_t'] for r in rs],[r['state']['phase_hz'] for r in rs],
            'o-' if branch['direction']==1 else 'o--',label=('旋向+' if branch['direction']==1 else '反旋向控制')+('，联合姿态' if branch['pose_refinement'] else '，仅相位'))
    ax.set_title('作者核心短跟踪：频率曲线不等于RPM真值');ax.set_xlabel('初始化后时间 / s');ax.set_ylabel('事件相位频率 / Hz');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'phase_inputs.png',dpi=150);plt.close(fig)
    table=''.join('<tr><td>'+case+'</td>'+''.join(f"<td>{score[case][arm]['error']['mean']:.3f}</td>" for arm in ['control176','isolated','pair_isolated'])+'</tr>' for case in ['side','train2'])
    timings=[r['elapsed_ms'] for b in pilot['branches'] for r in b['records'] if 'elapsed_ms' in r]
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>177—180：桨盘入口与局部相位</title><style>body{{font:16px/1.75 system-ui,"Microsoft YaHei";max-width:1050px;margin:32px auto;padding:0 20px;color:#173044;background:#f5f8fb}}h1,h2{{line-height:1.4}}section{{background:white;padding:22px;margin:18px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d8e2e9;padding:9px;text-align:left}}img{{width:100%;height:auto}}a{{color:#076b9d}}.note{{border-left:4px solid #cd8c23;padding:12px;background:#fff5df}}code{{overflow-wrap:anywhere}}</style>
    <h1>先把桨盘轮廓与真实旋转证据分清楚</h1><p>177—180 · 2026-11-07 · 原生ON/OFF、无NN、无坐标标注参与推理</p>
    <p class="note">本轮修复了孤立轮廓漏口，接通作者相位核心和单桨初始化；没有获得新的轨迹预测精度提升。默认仍为176，30px/持续RAW30FPS尚未达标。</p>
    <section><h2>卡点与完成状态</h2><table><tr><th>卡点</th><th>本轮处理</th><th>目前状态</th></tr>
    <tr><td>活动小核被当成整个桨盘</td><td>同一当前双极轮廓仅含一个已有核时，重拟合其原始外轮廓；保留核心来源、协方差、ROI/传感器边界检查</td><td>接口已修。43.6s的10.1px核→25.0px轮廓；不增加桨盘数。另一段10次触发，但0合格布局</td></tr>
    <tr><td>亮轮廓≠转速已测</td><td>检查真实严格周期票数；复用过去速度的事件平移；分离几何支持与周期支持</td><td>三个时刻每个轮廓原先仅0—2票；对齐后仍仅0—3票，达不到既有8票入口。没有把稀疏判据失败当作“桨叶物理上没有周期”</td></tr>
    <tr><td>完整五圆与相位初始化互相等待</td><td>单个观测椭圆即可做前半发现、后半验证；复用频率网格、空间二阶相位与有界峰值细化</td><td>C帧4个候选中1个通过，约263.58Hz事件相位；无须四个转速或完整模板。不等于机械RPM真值</td></tr>
    <tr><td>相位能否持续可靠</td><td>直接编译本机HelixTrack作者核心，独立C接口；自动轮廓/正相位先验，禁GT回退</td><td>已接通；C同旋向约258—265Hz，下一包预更新一致性仅.25—.41，未稳定过.5门槛，不能送入已知推力</td></tr>
    <tr><td>推力尺度、未知力学习与实时性</td><td>保留现有可见贡献＋未知项，作者支路不反馈推力</td><td>未解决。独立转速真值仍缺；不将隐藏桨叶补成实测；本轮未测持续RAW30FPS</td></tr></table></section>
    <section><h2>原生轮廓的具体变化</h2><img src="../isolated177_20261107/native_outlines.png"><p>黄色是174已有候选，青色是177孤立轮廓恢复。拒绝无核轮廓、多核合并、平线、裁边及不被当前原像素支持的膨胀。三桨尺寸/共面一致性仍不通过，没有降低门槛来制造第四桨。</p>
    <h2>预测对照：没有把候选恢复包装成精度提升</h2><table><tr><th>固定源时刻0.4s均误差 / px</th><th>176控制</th><th>仅孤立入口177</th><th>174双弧＋177</th></tr>{table}</table><p>侧视17/19个计划点可评分，另一段7/7。原生缓存实际GPU前端，三个分支共享同一GPU结果；推理禁止读坐标标注，输出后才评分。176控制位置与旧实际输出最大差0。都是历史开发片段，不能作为新盲测。孤立入口保持可选，未设为默认。</p></section>
    <section><h2>局部旋转证据与作者代码复用</h2><img src="phase_inputs.png"><p>单桨相位入口用现有120—960Hz、15Hz网格发现候选；峰值细化仅用发现半段，冻结频率/旋向再检查验证半段。阈值沿用现有空间相位约定：.5一致性、3:1旋向功率、至少9扇区，不按坐标评分调参。C候选3验证一致性约.648、相位衔接约.638；其余三个拒绝。几何来自同一完整窗口，因此不声称独立噪声概率界。</p>
    <p>编译使用外部 <code>D:\\drone forecast\\helixtrack-main</code> 作者头文件与已安装Eigen/OpenCV，仅项目内生成桥接DLL。作者仓库未改。桥接不加载RAW、YAML、HDF或真值文件。椭圆先在明确的弱透视假设下归一化，再交给作者逐事件相位/批次姿态核心；这是本地适配试验，不是作者完整初始化协议的复现。</p>
    <p>独立合成恒角速旋转：200Hz→200.002Hz；打乱时间后同样的点云拒绝。未来事件追加不改变结果，改变验证段不改变发现频率。原始活动较弱的43.6s候选仍被拒绝（验证相位一致性约.141），没有因为作者核心输出了频率数字就接入推力。直接用一种仅匹配椭圆的投影参数初始化会出现接近0Hz的静态锁定，负结果另存；弱透视归一化避免了这一种坐标参数误用，但没有解决真实弱信号问题。</p>
    <p>C片段桥接单桨单分支处理约{min(timings):.1f}—{max(timings):.1f}ms/包，含约1.25万环内事件；不包含原始解码、GPU滤波、几何提案、初始化频率搜索及全预测，不能换算成整个项目持续FPS。诊断的双旋向/仅相位对照也不作为生产的四支并行。</p></section>
    <section><h2>下一步按依赖关系做</h2><ol><li>先让每个当前桨盘的几何随时间更新，允许短时惯性续接与复锁；单桨相位不再等待完整五圆。每次记录输入证据失效原因，禁止固定一帧椭圆长期冒充当前姿态。</li><li>把通过当前窗口且下一包也一致的相位候选接入可见贡献；只形成可见项，隐藏项继续在未知力中。先查来源和持续性，不再先扫力增益/滤波时间常数。</li><li>有持续有效输入后，再做有/无该贡献的同起点0.4s消融；最后测完整RAW端到端30FPS。</li></ol><p>不能由一个C帧合格或短时频率曲线证明推力准确、泛化或创新成功。当前最大待办是自动逐桨更新与连续相位资格。</p></section>
    <section><h2>复用来源与数据文件</h2><p><a href="https://github.com/radimspetlik/helixtrack">HelixTrack作者仓库</a>：本地直接编译核心；<a href="https://github.com/Li-Zhaoxi/AAMED">AAMED作者仓库</a>：复查弧分组思路，本轮未复制其GPL代码；<a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Gallego_A_Unifying_Contrast_CVPR_2018_paper.pdf">Gallego等，CVPR2018事件对齐</a>：复用项目已有过去速度平移。作者许可尚需核对，桥接DLL未加入GitHub发布包。</p>
    <p><a href="../isolated177_20261107/integrated/scores.json">预测评分</a> · <a href="../isolated177_20261107/integrated/three_audit.json">三桨拒绝原因</a> · <a href="../alignment178_20261107/past_motion_alignment.png">平移对齐图</a> · <a href="../alignment178_20261107/aligned_period179.json">实际周期票数</a> · <a href="../helix_bridge180_20261107/ring_prior_checks.json">相位正负控制</a> · <a href="../helix_bridge180_20261107/review_C_ring_phases.json">C四桨相位资格</a> · <a href="../helix_bridge180_20261107/native_pilot_C.json">作者核心C短跟踪</a> · <a href="../helix_bridge180_20261107/native_pilot.json">弱信号片段负结果</a></p></section></html>'''
    (OUT/'index.html').write_text(document,'utf-8')
    files=['fast_structure_front.py','run_rotor_only.py','isolated_rim_visibility.py','observed_ring_phase.py',
        'helixtrack_adapter.py','native/helixtrack_bridge.cpp']
    files+=['tools/'+n for n in ['check_isolated_rims177.py','live_rims177.py','probe_geometry_alignment178.py',
        'probe_outline_seeds178.py','probe_aligned_period179.py','build_helix_bridge180.ps1','check_helix_bridge180.py',
        'pilot_author_phase180.py','check_ring_prior180.py','review_ring_prior180.py','report_rotor_inputs180.py']]
    manifest={}
    for n in files:
        p=OUT/'delivery_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
        manifest[n]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'delivery_manifest.json').write_text(json.dumps(dict(current_delivery_source=manifest,
        actual177_inference_snapshot='../isolated177_20261107/integrated/source_snapshot',
        author_headers_bundled=False,release_tree_modified=False,production_default='176',
        goal30px30FPS_achieved=False),indent=2),'utf-8')
    print(str(OUT/'index.html'))
