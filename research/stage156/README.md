# 151–156：按痛点优化后的当前交接

2026-10-24。推荐保留的研究组合：153 LM 求解、156 安全并行几何、无当前整机时跳过无效前端，以及154的未建模曲线回退。无开关默认仍是138＋140＋144。

报告与预测视频：`D:/drone forecast/output/stage156_parallel_geometry_20261024/分层预测优化与并行加速.html`。

## 运行

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' `
  'D:\drone forecast\event_drone_forecast\run_optimized_forecast.py' `
  --raw 'E:\dataset\train\6\media\event.raw' `
  --output 'D:\drone forecast\output\my_new_trial' `
  --start 110 --duration 9.5 --preview --preview-window-ms 5 `
  --pose-solver lm --parallel-geometry --skip-unobserved-geometry --curved-rollout
```

输出目录必须不存在。入口推理只读RAW和固定既有权重，不读旧预测或标签。
`optimized_geometry`保留同状态的原外推，`curved_rollout`为新的首选研究输出；
`preferred_forecast`明确指出选择。`valid=false`表示不发布，评分应计缺失。

## 实际结果

- 最终真实RAW：clear400、side475、train2 300，共1175时刻；前一版154也跑1175，不能当独立数据翻倍。
- 原历史系数可用口径：side0.4s 50.027→49.216px，66对/缺9；train2 59.301→58.930，30对/缺6。0.1/0.2s均小幅改善。
- **严格已发布口径**：side46.430→45.594px，64对/缺11；train2同上。不能把原50.027与新45.594跨口径作比较。
- clear仅未来检测器自洽参考，25.595→25.595。最终曲线回退在已建模状态沿用原预测；不是独立重心真值。
- 最终处理中位clear98.75、side59.84、train2 49.45ms，另有约11–14ms序列化/预览。仍不满足20ms实时。
- 153同输入交替模型中位10.85→8.75ms；156同输入100步几何64.18→60.32ms，观测和中心完全一致。跨批次整链路计时受系统/GPU负载影响，不把差值全部归因于代码。
- 最终各次RAW源指纹、事件时间截止、禁读标签/缓存通过。检测框保持149逐项相同；side/train2基线坐标差0；clear基线相对149最大差0.00426px（LM一次收敛改善）。

## 没有采用的路径

- 151 SDK稀疏光流：有独立速度，但加入KF后side0.4s 50.03→57.92、train2 59.30→62.88。只保存探针，未接主入口，不继续调权重。
- 152上一帧初始化：中心最大变23.7px、资格变10次，耗时仅省约2.6%；未采用。
- 154无条件曲线：清晰自洽25.595→25.814；拒绝普遍应用。156用原有结构资格决定回退，无视频名/标签规则。
- 155 OpenCV单线程：比现有16线程略慢，观测相同；未改变线程配置。
- 149惯性/相对尺度代码和四段RAW已经完成，但没有证明独立转速/推力收益。149早期handoff_verification149的RAW=0属于当时记录，不能覆盖后续完成文件。

## 并行与观测语义

156移除count/occupancy前端的进程全局检测函数切换，改为实例上的固定检测器。各臂拥有独立恢复、轨迹与模型状态，仅读共享的当前计数；同一时刻全部join后再预测。部件耗时是各臂之和，不能直接与几何墙钟耗时相加。

曲线资格是“同一目标是否已学成结构模型”，不是经过真值验证的正视/侧视分类；结构记忆也不是当前姿态。下一步应继续分开结构记忆与当前可观测性。

## 下一步与避免重复

1. 先减少前端成本与发布延迟；GPU本身约19–21ms，单步仍不能50Hz。应考虑低频几何与高频位置分工、最新观测队列及真实时间戳，不继续无约束FIFO积压。
2. 运动侧以156回退为新对照，只加明确机制。不要重做已失败的裸框斜率平滑、速度守门或按加速度模长分组调阻尼。
3. 桨盘研究先问“可靠姿态变化是否能提前提示机动”；与恒定方向、打乱时序对照。当前频率代理没有独立收益，不能直接叫推力，不能据弱相关断言真实物理信息无用。
4. 增加独立录像的少量机动片段参考，有价值的增量才扩跑。现有两个密集人工框集合合计111帧；0.4s旧口径96对，不是96段独立运动。

## 复用来源

- 已安装的 Prophesee [稀疏光流](https://docs.prophesee.ai/stable/algorithms/optical_flow.html)，151实际试用；未安装新依赖。
- SciPy [least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)，153使用现成MINPACK LM，不是新优化器。
- 既有本地138解析目标、107固定FilterPy数学、146–149观测模块。曲线运动/并行本身均不作为新颖性声明。

保护约束继续有效：不访问/修改yolov10-main；不使用G或灰度重建；E盘RAW只读。
