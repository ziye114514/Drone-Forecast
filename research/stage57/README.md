# 57：完整轮廓之后按整机框筛选桨盘候选

[实验结论与阈值诊断](</D:/drone forecast/output/stage57_candidate_roi_20261021/实验结论与密度阈值诊断.html>) · [三路数值、完整视频与固定时刻画面](</D:/drone forecast/output/stage57_candidate_roi_20261021/三路预测对照与固定时刻画面.html>) · [完整统计](</D:/drone forecast/output/stage57_candidate_roi_20261021/第五十七阶段完整轮廓候选ROI报告.html>)。

结果是修复55硬裁剪带来的清晰几何退化，尚未改善原方案的侧视预测，不升级默认。

| 0.4s轴向频率分支 | 原全图54 | 硬裁55 | 候选57 | 同有效/目标 |
| --- | --- | --- | --- | --- |
| clear6 | 14.22px | 17.20px | 14.22px | 2/3 |
| side6 | 122.46px | 120.94px | 122.45px | 5/6 |
| train3 | 136.69px | 136.69px | 136.69px | 11/12 |

联合成功数各段分别2/3、1/6、2/12，三种方案均未变化。侧视历史trusted_acc81.91px仍好于物理122.45px；新混合side约93.60px，原93.59px，无实际提升。评价是稀疏可见框未来端点，不是密集ADE、重心或姿态/RPM精度；开发已见数据不是新盲测。

## 改动范围

实际复用原9/8完整全图计数、非零密度分位数、形态学及轮廓拟合；CandidateFrontend仅在候选拟合完成后、原排名/分组前按其中心筛选。框内椭圆即使周界越过框界仍保留完整字典与形状。25%余量、160ms期限复用55 GeometryROI，提示不晚于20ms包开始，缺失或过期退全图。没有删除当前事件、改变频率历史、制造缺桨观测或调低机架门槛。

原始完整频率/相位/关联/RLS和15模型保持。原生20ms整机像素/框/位置与六历史及有效覆盖固定，关闭完整输出须精确54。run_candidate_forecast.py默认是本实验候选模式；已验证的原完整基线run_deploy_forecast.py保持。

## 真正定位到的副作用

用原8 AST仅增加一条记录，两个独立RAW进程各200时刻完整几何精确55全图/硬ROI。192个启用ROI时刻阈值全部变化，每帧阈值比值中位1.315、最大1.503。全段阈值中位3.025→3.977；非零密度格中位18412→11631。少量背景事件的删除改变了全局分位数，不能假定“只是加框、检测参数没变”。

新方案恢复清晰四候选及频率，原机架合格时刻3→60、最长连续1→13。但阈值变化与轮廓切边尚未单独做因子消融，不称阈值为唯一原因。

侧视原机架合格2/500、最长连续2，3号0/600。旧仿射平行四边形假设可能被透视和椭圆中心偏移破坏，四点任意单应性也不能证明真实四桨身份。

## 复核范围

主体13个case共1715父/3430原生10ms时刻、1480视频帧全解码，真实RAW前缀225/450全部字段哈希、ROI/候选计数、完整15预测与生命周期精确；运行内完整字节SHA仍保留。360未来评分行全部覆盖一致，252基线/历史行精确54，空场零输出。三个pilot批次35676、正式20880、评分/作图/探针27526均退出0；没有本轮IPC失败。阈值探针额外400原生10ms观测单独计数，不算新的完整物理预测实验。

冻结源码、派生audit/prefix/score、模块检查、delivery_supplement及density_probe_supplement均保留。派生文本冻结用LF，Windows磁盘文件可能CRLF；核对源文本须仅规范换行，ZIP按实际字节校验。

## 单独运行

新输出目录必须不存在，不覆盖固定实验结果：

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_candidate_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --start 108 --duration 5 --output 'D:\drone forecast\output\my_candidate_trial' --geometry-roi body --precision 32 --verify-frames --preview
```

改为--geometry-roi off是原全图控制。单步仍远超20ms，不能由50fps源时间视频推断实时。

## 下一步与已有工作

已有[事件桨叶状态估计](https://arxiv.org/html/2604.18289v1)和[EVPeriscope](https://arxiv.org/html/2609.11920v1)的检测、ROI与跟踪基础；本轮是本地模块受控整合，不称新颖算法或作者完整复现。

58优先核查四个共面桨盘的圆锥一致性、中心投影与消歧，补充四点任意单应性缺少的独立约束。原始研究[共面圆锥标定及歧义分析](https://link.springer.com/article/10.1186/s41074-018-0050-y)讨论共同单应性与多解；[圆心校正TIM2026作者库](https://github.com/xiahaa/circular-center-calibration)已公开2D中心/多圆消歧代码，给定内参的合成协议见[作者实验说明](https://github.com/xiahaa/circular-center-calibration/blob/main/docs/experiments/synthetic_2d_accuracy/README.md)。它不是事件无人机预测器，未知标定和残弧条件仍须验证。

[ICRA2024直接模型事件追踪作者库](https://github.com/BluecatLi/direct_event_object_tracker)需3D模型、ROS/Linux及初始位姿设置，不能直接替换当前RAW自动入口。上述新库本轮只做资料核查，没有安装或执行。下一轮先固定版本/许可、检查最小2D依赖，在49已知几何中验证多解、噪声和侧视失败，再决定真实当前观测试验；不得把合成精度当自采真值。
