"""Update handoff documents only. This script starts no experiments or workers."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUTPUT = PROJECT / 'output'

status = '''# 项目收尾交接：2026-10-19

**按用户“把当前工作收尾后就结束”暂停。最新完成阶段48；阶段49尚未启动，无待运行任务，无本项目Python/FFmpeg后台进程。用户明确继续后再启动实验。** 长期研究目标尚未完成。

先看[实验总导航](</D:/drone forecast/output/research_overview_20261018/index.html>)、[完整WORKLOG](</D:/drone forecast/WORKLOG.md>)和[最近交接清单](</D:/drone forecast/output/checkpoints/latest_handoff.json>)。历史记录的“运行中/active”已被本状态和WORKLOG末尾更新。

## 最新完成

- 43：真实耗时拆解，几何中位26–43ms、网络内部约43ms、PNG编码6–8ms。100时刻完整观测/预测与42精确一致。
- 44–45：复用官方ONNX Runtime CUDA Graph及固定地址IO绑定。45同进程模型图像链PyTorch47.77ms、FP32 Graph22.86ms。1550帧重新推理后，框数及各时距六模型覆盖/联合成功统计保持，仅有微小浮点差。
- 46：标准FP16转换及1550帧完整回归。配对模型链FP32 Graph27.67ms、FP16 Graph21.96ms，离散质量统计保持。cuDNN初始化有兼容提示，原失败及补充修复协议均保留。独立连续运行10.16ms与配对交替计时组织不同，不能混算加速倍数。
- 47：共享内存RAW入口，390时刻逐像素校验、独立前缀一致性和350视频帧解码通过。完整几何FP32单步中位74.52ms；关闭桨盘几何、仅整机位置的FP16为31.05ms，84%时刻仍超过20ms。处理计时不含RAW下一块读取/写盘/预览，尚未实现完整实时预测。
- 48：更准的整机位置上重新比较轴向、频率平方、在线推力和延迟加权，700时刻、三种转子处理。六历史模型系数回归通过，预测先保存再读人工未来框。这是缓存同状态消融，尚未合并为新RAW完整物理入口。

## 效果与卡点

整机微调的3号录像107个有机时刻106检出、105个IoU≥0.5；连续段当前覆盖3/12→12/12、0.4秒预报覆盖3/12→11/12。该录像此前开发见过，属于权重留出，不能称全流程盲测；可见框中心不等于真实质心，没有实飞姿态、机械RPM或推力真值。

48同位置0.4秒条件端点误差：清晰段CV14.31→轴向+频率14.22像素(仅2/3目标)；侧视124.93→124.36，仍差于收缩历史模型81.91；3号137.05→136.69，仍差于历史模型110.32。**物理信息尚无稳定额外收益，复杂转弯未来误差仍大。** 各短段不能混成统一精度，需同时阅读覆盖和共同样本。

具体原因之一已定位：侧视新换槽触发20次物理重置、3号10次；模型有0.6秒预热，侧视新方案与3号实际参数更新均为0，清晰段58次。3号还受四桨质量门槛限制。原`completion.methods.updates`未递增，不能当测量值；真实计数已从逐帧模型状态复算到[修正统计](</D:/drone forecast/output/stage48_body_rotor_physics_20261019/corrected_learning_summary.json>)，原结果保留。

## 可运行入口与范围

- 最新位置/历史入口：[run_shared_body_forecast_v2.py](</D:/drone forecast/event_drone_forecast/run_shared_body_forecast_v2.py>)，用法和视频见[47报告](</D:/drone forecast/output/stage47_shared_raw_bridge_20261019/第四十七阶段共享内存RAW入口报告.html>)。v1因代理漏PIPE常量在创建GPU子进程前失败，已保留；应使用v2。`--body-only --precision 16`明确关闭桨盘几何，不是完整姿态/推力系统。
- 完整桨叶物理RAW路径仍为[run_adaptive_physics.py](</D:/drone forecast/event_drone_forecast/run_adaptive_physics.py>)，见[30报告](</D:/drone forecast/output/stage30_raw_adaptive_physics_20261018/第三十阶段RAW自适应物理预测报告.html>)。48不等于新在线入口已集成。
- SDK环境`D:/drone forecast/.venv-event311/Scripts/python.exe`，GPU子进程为本机Python313。没有新依赖安装、额度重置、外部消息或自动续跑。

## 下次明确继续后的第一步

先修复“频率参考改变就清空全部学习”：保留与频率无关的轴向模型状态，再研究频率归一化参考变化时的等价参数、协方差和队列变换。先检验代数不变性，再冻结与48相同输入/未来目标的对照；不要同时放宽门槛造成归因不清。**仅为待办，阶段49未启动。**

在线RLS和轨迹残差估参已有研究，不能直接称新颖：继续核查[事件桨叶相对状态估计](https://arxiv.org/abs/2604.18289)、[Fast RLS Identification](https://arxiv.org/abs/2508.17577)、[Motor Efficiency via Residual Minimization](https://arxiv.org/abs/2510.11388)。没有证明前，不声称恢复真实质量、风速、推力系数或超过论文。

源码/协议/小型报告已建立交接快照，RAW、权重、视频和完整JSONL仍在原目录，不包含在ZIP。旧默认、旧src、冻结结果和失败全部保留；不用G通道，不访问/执行/修改`D:/yolov10-main`。
'''
(PROJECT / 'PROJECT_STATUS_20261019.md').write_text(status, encoding='utf-8')

log = '''

## 54. 已完成：47共享内存RAW入口与核验

- 47原代理缺subprocess.PIPE，在首个worker创建前失败；保留冻结v1与failed_train3_full32_pipe_proxy_v1。新run_shared_body_forecast_v2.py只补常量及接口检查，repair47/pipe_proxy_supplement冻结并完成，工具15846已结束。
- 五case共390原生时刻父子像素SHA与41参考逐像素一致；完整32位几何100时刻精确回归，NN结果对应45/46。独立40时刻前缀六预测精确，空场50当前/预报全零，四视频350帧全解码。worker已wait回收、共享内存已释放。
- 完整几何32位单步74.516ms/P95 90.961；body-only32位38.712ms；body-only16位31.049ms/P95 41.886，84%超过20ms。计时排除RAW下一块/写盘/视频，wall含冷启动。不同日42约102ms不能直接算严格加速；body-only关闭几何，不能冒充完整物理链。报告、regression与全部失败保留。

## 55. 已完成：48新整机位置上的桨叶物理贡献

- compose48/study48冻结；同45 body_first原生20ms位置接34真实10ms传感器同截止时刻：clear100、side250、train3 300、negative50，共700。legacy_rotors/slotfix_rotors/associated_slotfix三路；关联仅用当前扩大25%的整机框，不伪造缺失桨频。28轴向、27推力RLS、29延迟15专家均实际执行。六历史模型系数/有效性与45逐项相同，预测先保存再读粗框评分。工具6453正常结束。
- 清晰0.4s CV14.30775→轴向频率14.22397px(2/3)，侧视CV124.933→新频率124.361仍不及trusted81.913，train3 CV137.049→136.690仍不及trusted110.319(11/12)。没有稳定物理额外收益；负例零输出。本阶段为缓存同状态消融，不是新RAW在线物理集成。
- 侧视新换槽20次重置，train3 10次。0.6s预热与四桨质量门槛限制实际学习：clear58次、side legacy1/新0、train3均0。completion.methods.updates只是未递增初始化占位，实际从逐帧feature_diagnostic.models计数复算到corrected_learning_summary.json。原冻结结果保留，报告明确修正。
- 已查原始论文2604.18289、2508.17577、2510.11388：桨叶状态估计、在线RLS、残差估参都已有先例。下一步建议仅是频率参考改变时保留轴向状态及做等价增益/协方差/队列坐标变换，先代数不变性、后48同输入消融；尚无阶段49代码或实验。

## 56. 2026-10-19按用户要求收尾并停止

- 用户明确“把当前工作收尾后就结束”。目标保持paused，长期目标未标完成；不启动49，不自动续跑，不再请求额度或调用重置。
- 只读Win32_Process检查未发现本项目Python/FFmpeg进程；所有已知实验句柄均结束。整理PROJECT_STATUS_20261019.md、README与总导航到48，保存源码/补充协议SHA及HTML链接检查、交接ZIP和清单。更早active/运行中为历史，以上述最新状态为准。
- 原数据/权重/视频/JSONL及失败保留，ZIP仅源码和小型汇总。下次用户明确继续后先阅读最新状态和本日志54–56，不重跑已有实验覆盖结果。
'''
worklog = PROJECT / 'WORKLOG.md'
old = worklog.read_text(encoding='utf-8')
if '## 56. 2026-10-19按用户要求收尾并停止' not in old:
    worklog.write_text(old + log, encoding='utf-8')

readme = ROOT / 'README.md'
old = readme.read_text(encoding='utf-8')
end = old.index('2026-10-17第九阶段：')
head = '''# Event Drone Forecast：只输入RAW的二维无人机预测工程

2026-10-19按用户要求收尾暂停，已完成至48，无后台任务，不自动续跑。先读[最新交接状态](</D:/drone forecast/PROJECT_STATUS_20261019.md>)、[实验总报告](</D:/drone forecast/output/research_overview_20261018/index.html>)和[WORKLOG](</D:/drone forecast/WORKLOG.md>)。以下较早记录和旧默认保留。

最新可运行位置入口是[run_shared_body_forecast_v2.py](</D:/drone forecast/event_drone_forecast/run_shared_body_forecast_v2.py>)：原生ON/OFF、固定适配权重、共享内存及常驻本地GPU；用法/视频见[47](</D:/drone forecast/output/stage47_shared_raw_bridge_20261019/第四十七阶段共享内存RAW入口报告.html>)。完整几何FP32单步74.52ms，仅整机位置FP16为31.05ms，两者仍超过20ms预算；后者关闭桨盘几何，不等于完整物理链。原v1失败保留，使用v2。

整机检测明显改善，未来位置仍有较大误差。46完整1550帧回归保持覆盖和联合成功；48在同一更准位置上加入桨叶物理，尚无稳定额外收益。侧视换槽20次重置导致参数实际更新0，是下一步应优先处理的瓶颈，阶段49未启动。没有实飞姿态、机械RPM或推力准确性证明。

完整桨叶物理RAW路径仍为[run_adaptive_physics.py](</D:/drone forecast/event_drone_forecast/run_adaptive_physics.py>)，见[30](</D:/drone forecast/output/stage30_raw_adaptive_physics_20261018/第三十阶段RAW自适应物理预测报告.html>)。48为缓存同状态消融，新整机位置尚未与完整物理路径合并为已验证更好的在线系统。不用G，不访问/修改yolov10-main；旧run.py、42入口及冻结失败保留。

'''
readme.write_text(head + old[end:], encoding='utf-8')

overview = OUTPUT / 'research_overview_20261018/index.html'
page = overview.read_text(encoding='utf-8')
lead = '''<div class="lead"><b>2026-10-19已按用户要求收尾暂停，完成至48，无后台任务。</b><p>整机检测改善，未来预测仍有较大误差。最新RAW入口47：完整几何FP32单步74.52ms，仅整机位置FP16为31.05ms；两者尚未达到20ms预算。后者关闭桨盘几何，不能当成完整姿态/推力系统。</p><p>46的1550帧回归保持覆盖/联合成功。48同位置物理消融未显示稳定额外收益：侧视换槽20次重置，参数实际更新0。下次优先检查参考变化时保留学习状态，阶段49尚未启动。3号录像此前开发见过，不是全流程盲测；稀疏可见框不等于真实质心、姿态或RPM。</p></div>'''
page, count = re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)', lambda _: lead, page, count=1, flags=re.S)
assert count == 1
cards = '''<article><h2><a href="../stage48_body_rotor_physics_20261019/第四十八阶段整机位置与桨叶物理贡献报告.html">48 · 新整机位置与桨叶物理贡献</a></h2><p>700时刻同位置消融；侧视重置20次、实际参数更新0，物理信息尚无稳定额外收益。</p></article><article><h2><a href="../stage47_shared_raw_bridge_20261019/第四十七阶段共享内存RAW入口报告.html">47 · 共享内存RAW入口与预测视频</a></h2><p>390时刻校验；完整几何74.52ms，仅位置FP16为31.05ms，尚未实时。使用v2入口。</p></article><article><h2><a href="../stage46_fp16_graph_20261019/第四十六阶段半精度图执行验证报告.html">46 · FP16与完整质量回归</a></h2><p>1550帧覆盖/联合成功保持；同进程FP32/FP16模型链27.67/21.96ms，非完整延迟。</p></article>'''
if '../stage48_body_rotor_physics_20261019/' not in page:
    page = page.replace('<div class="grid">', '<div class="grid">' + cards, 1)
page = page.replace('最新继续状态</a>', '最新收尾交接</a>')
overview.write_text(page, encoding='utf-8')
print('Updated paused handoff, WORKLOG, README and overview through stage48; no experiments started.')
