"""Persist completed records and next actions without claiming goal success."""
from pathlib import Path
from datetime import datetime,timezone
import json
ROOT=Path(__file__).resolve().parents[1];project=ROOT.parent
checkpoint=project/'output/checkpoints/latest_handoff.json'
data=json.loads(checkpoint.read_text('utf-8'))
data.update(utc=datetime.now(timezone.utc).isoformat(),latest_numbered_stage=165,
    execution_status='Stage165 in progress: native correlation preprocessing and partial-entry velocity initialization retained; experimental plane LK and new force priors not promoted.',
    current_status_document=str(ROOT/'docs/HANDOFF_20261104_NATIVE165.md'),
    latest_report=str(project/'output/native_plane165_20261104/index.html'),
    next_action='Continue rotor-native mainline: complete current RAW axis-workspace test, then address actual decode/GPU handoff waits and carry a small independent recording; no quota10% threshold, no coupons. Do not repeat rejected LK, exposure timing or Matern priors as new ideas.')
data['stage165_progress']=dict(directory=str(project/'output/native_plane165_20261104'),
    user_resumed_on='2026-11-04',goals_achieved=False,
    retained=['native boundingRect instead of whole-image coordinate arrays','OpenCV sigma1.5 reflected Gaussian for native matching','partial-entry-only two-complete-position initialization','avoid unused CPU event copy after native GPU front; all native events still processed'],
    error_warm_fixed_px=40.35609951108299,error_warm_previous_px=44.10575381606213,
    error_warm_extension_px=44.423644027093175,extension_available=8,extension_eligible=10,
    error_cold_fixed_px=38.64692406948676,cold_available=9,
    latest_completed_cold_RAW_FPS=24.25576344224477,
    not_promoted=['LK shared affine/line transform: warm46.88px, slower','coarse matching: same predictions but no speed gain','uniform exposure lag: mixed regressions','Matern force derivative: cold40.48/warm47.07/extension50.58','mixed gain permission:0 updates due short/no excitation'],
    pending='actual RAW test of exact integer side-axis workspace; report is being assembled')
checkpoint.write_text(json.dumps(data,ensure_ascii=False,indent=2),'utf-8')
log='''

## 2026-11-04 阶段165进行中：继续攻主线痛点

- 用户恢复优化，要求先判断机制是否针对痛点，继续到额度耗尽；不恢复旧10%阈值，不用重置券/购买额度。无子代理、无新训练，RAW/保护目录/权重/发行ZIP未改。
- 重读164交接后发现根latest_handoff仍停163，现同步165进行中。此前164报告HTML尚未生成，不能拿不存在的链接当成已交付。
- 读到新论文CoHelix(arXiv2610.31384)：四桨共享投影、各自相位/速度/加速度，与共同平面方向相关；需要逐桨环标注/旋转方向/初始速度，原文未找到可直接使用的CoHelix代码链接。只借鉴思想，不声称已复现或五圆共享几何首次创新。
- 实装当前桨叶LK+共同仿射/侧线相似变换，复用旧OpenCV/LK资格。实际暖片80次资格，但固定9误差44.11→46.88、后续44.42→46.27，变慢至18.59内存FPS；未纳入推荐。侧视重复旋转纹理的当前对应仍不稳，不能靠换成仿射自动解决。
- 分段计时明确运动匹配和几何各约13–15ms。粗搜+原分辨率精定位不改变本片预测但不提速，保持可选。实际热区还含整幅np.nonzero与SciPy模糊：改C++boundingRect和同σ1.5/radius6/反射边界OpenCV Gaussian。暖片运动平均13.4→7.0ms，0.4s误差变化<0.001px，小收益保留。
- 边缘入画后的0初速先验会把已有高速误当大加速。--rotor-entry-init只在先有暂定/截断观测、160ms内首次完整定位时，使用前两次完整位置初始化速度/协方差；普通冷启动沿旧状态。直接RAW108–114.4s：固定9误差44.106→40.356px，P95约58.71；后续10查询可评分8个仍44.424px，不冒充独立录像。193包/118检测/119有效源预测，完整RAW18.96FPS，含空场/出画，尚未30。
- 冷启动RAW110–112s当前38.6469px，9/9；61包/60区域/57有效源预测/48代理输入。去掉GPU全量处理后未消费的CPU原生事件副本，并用规范16字节ABI的整块字节复制：完整24.26FPS，没有证明单项加速。GPU仍处理全部原生事件；只在native GPU+空间相位关闭+预取时适用，--rotor-retain-host-events保留控制。
- 继续试的物理机制：曝光中心延迟假设冷41.02px、暖46.47、后续41.40(混合退化)；Matérn3/2净力导数状态沿用共享可测/不可测分账，冷40.48、暖47.07、后续50.58；都不接默认。数学机制/协方差/分账检查通过不等于真实精度改善。开放局部混合代理增益学习仍0次更新，激励/连续来源不足，不冒称学出推力系数。
- 当前新增原生侧轴工作区，复用LSD对象、整数ON/OFF的同一log映射与相同行主序取样，不需要与NumPy2.4不兼容的旧Numba。8个真实保存输入采样坐标相同、0..4095整数映射相同，当前正在跑完整RAW实际速度/输出检查；源码守卫完成前不编辑运行依赖。
'''
with (project/'WORKLOG.md').open('a',encoding='utf-8') as f:f.write(log)
print('Progress165 saved')
