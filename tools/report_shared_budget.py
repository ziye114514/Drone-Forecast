from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,html
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/shared_template_budget_20261027'
def read(p):return json.loads(p.read_text('utf-8')) if p.exists() else None
def num(x):return '—' if x is None else f'{x:.2f}'
def main():
    frozen=WORK/'output/github_release_final_20261027/event-drone-forecast'
    same={p.name:p.read_bytes()==(frozen/'src/event_drone_forecast/position'/p.name).read_bytes() for p in (ROOT/'src/event_drone_forecast/position').glob('*.py')}
    assert len(same)==7 and all(same.values())
    digest=hashlib.sha256((frozen.parent/'event-drone-forecast-v0.4.0-alpha.1.zip').read_bytes()).hexdigest()
    assert digest=='ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'
    runs={}
    for name in sorted(p.name for p in OUT.iterdir() if p.is_dir() and (p/'protocol.json').exists()):
        s=read(OUT/name/'summary.json')
        if s:runs[name]=dict(summary=s,accuracy=read(OUT/name/'shared_motion_accuracy.json'))
    folder=OUT/'latest_source_snapshot';folder.mkdir(exist_ok=True);sources={}
    source_names=set()
    for name in runs:source_names.update(read(OUT/name/'protocol.json')['sources'])
    for name in sorted(source_names):
        data=(ROOT/name).read_bytes();p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);sources[name]=hashlib.sha256(data).hexdigest()
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),goal_achieved=False,runs=runs,sources=sources,
        component_counts=read(OUT/'fused_count_probe/summary.json'),component_conics=read(OUT/'fused_conic_checks/summary.json'),
        failed_or_unpromoted={n:read(OUT/n/'summary.json') for n in ['unknown_force_probe','coupled_goal_probe','raw_chunk_probe','flow_gain_result']},
        mechanism_pilots={n:read(OUT/n/'summary.json') for n in ['flow_force_balance_probe','axis_amplitude_probe','anchor_velocity_probe','native_latest_checks','sdk_activity_probe','anchor_amplitude_controls']},
        precommitted_8s_extension=read(OUT/'side_extension8s/extension_summary.json'),
        latest_visibility_reanchor=read(OUT/'side_visibility_final8s/extension_summary.json'),
        current_probes={n:read(OUT/n/'summary.json') for n in ['fused_line_checks','rigid_side_probe','rigid_current_CI_probe','visibility_anchor_probe','visibility_reanchor_controls']},
        position_core_verbatim=same,published_ZIP_SHA256=digest,scope='Exposed development snippets, no heldout/SOTA/RPM/thrust claim',
        next='Side final47.21px/36.88capacityFPS, clear GC30.41capacityFPS; early bursts still cause backlog, train2 remains60px. Adaptive period execution and equivalent NN preprocessing next; no label tuning or ROI sweeps.')
    (OUT/'batch_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    table=[]
    for name,v in runs.items():
        s=v['summary'];a=v['accuracy'];e=None if not a else a['scores']['RAW_current_motion']['physical_future_error_px']['0.4']['mean']
        table.append(f'<tr><td><a href="{name}/summary.json">{html.escape(name)}</a></td><td>{s["frames"]}</td><td>{s["fresh_image_observations"]}/{s["valid_forecasts"]}</td><td>{num(s["steady_capacity_fps"])}</td><td>{num(s["paced_fresh_observation_rate"])}</td><td>{num(e)}</td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>共享五圆计算与当前运动校正</title>
<style>body{max-width:1120px;margin:40px auto;padding:0 24px;background:#f4f6fb;color:#152338;font:17px/1.7 system-ui}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #d4dce9;padding:10px;text-align:left}code{background:#e6ebf4;padding:2px 5px}a{color:#1265a6}.note{border-left:4px solid #e3a321;padding:10px 20px;background:#fff9e8}</style>
<h1>共享五圆计算与当前事件位移校正</h1><p>目标：0.4秒平均位置误差低于50px、完整RAW流水线达到30FPS且不持续积压。当前尚未全面达成。下列均为已经使用过的开发片段。</p>
<h2>最新联合配置</h2><p><b>8秒实际RAW：26目标平均43.27px、P9580.60px、16/26小于50px。原110–114秒43.74px，新114–118秒42.23px；全部240源、174当前观测、170有效预测保留。</b>原4秒有目标段35.41处理FPS，清晰段32.94FPS。初期最大延迟约610ms，后续恢复；不能称从第一帧起稳定低延迟。全8秒50.12FPS包含很多无目标时刻，不当作检测帧率。</p><p>修复可靠框从局部可见到更完整时的中心基准跳变，结合当前轨迹、切向未知力和结构指导NN刷新。原始同输入二力控制74.30px，只校正运动45.20px，同窗二力44.50px，再加切向反馈43.27px。train2新实测仍62.65px且周期输入0，整体目标仍未完成。</p><img src="side_visibility_final8s/prediction_and_latency.png" style="width:100%" alt="最新8秒预测误差及延迟"><p><a href="side_visibility_final8s/extension_summary.json">最新分段明细</a>；下方保留各轮历史结果，不混作同一配置。</p>
<table><thead><tr><th>配置与结果</th><th>源时刻</th><th>当前观测/有效预测</th><th>处理能力FPS</th><th>当前观测Hz</th><th>0.4s均值px</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table>
<p class="note">处理能力为每帧总服务时间均值的倒数，不等于稳定低延迟观测率。启动单列并包含JIT导入/编译；所有RAW源时刻保留。异步NN时序会改变观测，跨次误差不能全归因某个算法。清晰26–28秒没有独立人工框。</p>
<h2>实际改动</h2><ul>
<li>融合20ms原生ROI过滤和ON/OFF计数，没有采样或截断；三个固定窗口逐像素一致，清晰同步60帧预测保持一致。</li>
<li>椭圆不确定度仍用同样10个中心差分点与步长，仅融合距离循环；40个受控椭圆和三个原生窗口数值差为0。</li>
<li>五圆固定模板设计矩阵只建一次，移除重复数组拼接。厚度/多尺度必要条件失败后跳过LSD，省掉的线诊断标未知。</li>
<li>复用定位已计算的双向KLT位移，以协方差交集修正速度、过去合格位移求总加速度；没有新光流遍历或位置KF。这是图像运动反馈，不是桨叶推力测量。</li>
<li>10万/100万RAW读取块输出字节一致，但100万这次更慢，默认保留10万。</li></ul>
<h2>最新收益与卡点</h2><p><b>最新侧视实际RAW：0.4s平均47.21px，P9583.84px，50px内9/18；处理能力36.88FPS，当前观测28.40Hz，120源/114当前/110有效。末10完成延迟12–27ms，但初期突发有最高数百毫秒积压。清晰新短测30.41处理FPS、28.82当前Hz，仍有积压。</b></p><p>同一最新输入：不复用位移的原二力63.31px、只复用校正位移48.28px、同窗二力残差47.21px。弱框升级强NN框时，框中心校正不应算成真实速度：新增现有事件轨迹的速度资格，位置和有效输出保持，主要修复了一个初始化大误差，未证明所有变向都改进。</p><p>早期阶段同输入62.42→58.50px仍保留；旧缓存47.08px用了2px代理，不能代替实际RAW结果。最新47.21来自真正新RAW及真实散布。</p>
<p>train2评分36个人工锚点都有观测。局部放大尝试7次、0新增合格框；无周期输入段的总加速度仍不准。未知力反馈和耦合观测器都退步，未启用。</p>
<p><b>冻结候选后的8秒扩展：</b>预先固定原110–114秒仍47.60px；新114–118秒65.71px，整段26目标53.17px。原范围没有移动，扩展没有过50px。整段50.28FPS包含很多无目标的轻负载时刻，不当成检测提速；原4秒约35.89处理FPS，积压最大仍561.65ms、末10平均20.57ms。<a href="side_extension8s/extension_summary.json">分段明细</a>。</p>
<h2>预测效果</h2><p>下面按冻结结果绘图，灰线为同输入控制，蓝线为本轮候选。视频黄色为预测、红色为人工未来参考；参考仅用于离线显示。30Hz视频播放速度不等于处理FPS。</p><img src="side_final_goal/prediction_and_latency.png" style="width:100%;background:white" alt="同输入预测误差与延迟"><video controls preload="metadata" src="side_final_goal/prediction_preview.mp4" style="width:100%"></video>
<h2>复现与交接</h2><p>聚合入口<code>--target-candidate --paced</code>；侧视47.21px/36.88FPS那次为<code>--goal-profile --jit-conics --flow-force-balance --axis-amplitude-bridge --defer-cyclic-gc --paced</code>。清晰30.41试验另加--parallel-period。EDF_NUMBA_PATH指向隔离依赖目录。首次导入/编译另计启动；最新生产计时关闭--verify-frames，早期功能核验开哈希，不把关调试当算法收益。</p><p>可选--sdk-avx2调用已安装SDK的本地AVX2编译实例，三个组件过滤约减半、完整60帧预测逐项保持；不代表完整系统翻倍。SDK头文件和本地二进制不进入发布包。原SDK过滤定义见<a href="https://docs.prophesee.ai/stable/api/cpp/cv/algorithms.html">Prophesee官方文档</a>。</p><p>周期线程/独立进程、共享输出、批量最新像素索引、二力局部增益的所有正负结果保留。局部增益真实数据0次合格更新，不能宣称未知推力系数已经学出。GC只在有限回放流内延后，引用计数释放不变，结束恢复；尚未验证无限流内存稳定，未审计对照的events=[]也不代表没有GC暂停。</p>
<p><a href="batch_summary.json">全部数据与源码哈希</a> · <a href="side_shared_motion/shared_motion_accuracy.json">侧视同输入评分</a> · <a href="fused_count_probe/summary.json">计数组件</a> · <a href="fused_conic_checks/summary.json">差分组件</a> · <a href="raw_chunk_probe/summary.json">读取块试验</a></p>
<p>原发布ZIP与原位置核心7文件逐字节保持。本轮只改本地工作项目，不上传或替换GitHub包。可选依赖依据<a href="https://numba.readthedocs.io/en/stable/user/installing.html">Numba官方兼容性文档</a>。</p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    path=WORK/'output/checkpoints/latest_handoff.json';h=read(path)
    h.update(utc=result['updated_utc'],execution_status='Active optimization; read current batch artifacts before continuing',research_objective_status='incomplete')
    h['shared_template_budget']=dict(report=str(OUT/'index.html'),summary=str(OUT/'batch_summary.json'),goal_achieved=False,
        side_actual_error_px=47.21444357525477,side_same_inputs_control_px=63.305702828356374,
        clear_latest_actual_fps=32.93747144901228,clear_adaptive_fps=29.987038991474012,
        clear_best_short_fps=32.93747144901228,side_latest_actual_fps=35.40723738079758,
        side8s_error_px=53.1706934152365,side_extension_only_error_px=65.71498753738433,
        side8s_fixed_original_error_px=47.59545158317079,side8s_original_window_capacity_fps=35.88965601641012,
        early_backlog_and_train2_goal_incomplete=True,next=result['next'])
    h['shared_template_budget']['latest_visibility8s_mean_px']=43.27445201950333
    h['shared_template_budget']['latest_train2_mean_px']=62.64638863671062
    h['quota_continuation_policy']=dict(latest_user_instruction='Continue until quota exhaustion; supersedes10percent clock-only threshold',
        checkpoint_each_short_batch=True,no_credit_purchase_or_reset_coupon=True,do_not_reuse_past_reset_clock_times=True)
    path.write_text(json.dumps(h,ensure_ascii=False,indent=2),'utf-8')
    log=WORK/'WORKLOG.md';text=log.read_text('utf-8');marker='## 2026-10-28 共享计数、五圆重复计算与当前事件位移反馈'
    if marker not in text:
        text+='\n\n'+marker+'''
- 最新用户授权持续做到额度耗尽，覆盖此前10%时钟等待规则；不购买/重置额度。目标active、未完成，按短批保存。
- NumBa0.64/LLVM0.46隔离于.cache/numba_cpu_064，SDK NumPy2.4.6不变。融合ROI+ON/OFF计数三原生窗逐像素一致；40椭圆+三窗同10差分不确定度最大差0。清晰同步60预测与旧输出相同。
- 清晰goal+融合计数28.3836FPS；复用真实位移散布+差分融合另次25.1873FPS、59当前55有效，不能跨异步输入全归因单模块。侧视120源114当前110有效、31.3387处理FPS、28.3773当前Hz；初始积压，末10延迟17–52ms。
- 复用现成KLT，无新光流遍历/位置KF。实际侧视同输入0.4s62.4239→58.5046px(18目标)，0.1s12.3253→9.6855，保留小收益；早先2px代理缓存47.0843不能称实际RAW达标。
- train2 currentNN111当前107有效、36/36人工锚点检测，0.4s62.885。ROI7次0接纳、61.4949变化不能归因恢复。未知力反馈side52.668/train279.224、耦合观测器side88.971/train295.717均退步，不启用。
- RAW读取10万/100万原生与过滤字节一致，100万更慢，默认仍10万。新剖析定位重复np.c_和固定设计矩阵，已缓存/column_stack/concat；厚度/多尺度必要门槛失败才跳LSD，诊断标未知。后续完整结果见batch_summary。
- 报告output/shared_template_budget_20261027/index.html及latest_source_snapshot；发布ZIP与原位置7文件保持。不要宣称50px/稳定30FPS/真实推力/独立泛化已完成。
'''
        log.write_text(text,'utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),runs=list(runs),position_files_preserved=7,archive_unchanged=True)))
if __name__=='__main__':main()
