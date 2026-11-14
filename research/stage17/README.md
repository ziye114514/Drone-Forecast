# 阶段17：周期证据、桨盘几何与ROI

2026-10-18。自采RAW四段35秒/1750时刻固定小实验。当前结论：**不替换旧默认前端**。清晰8/8稀疏框有发布；侧视9/14，低于旧前端按同标签重评的13/14；train4为3/5；train5无机0/400发布。清晰ROI中位122.88ms，未实现50Hz实时。无未来轨迹或真实RPM提升结论。

报告：[第十七阶段桨盘几何与ROI实验报告](</D:/drone forecast/output/stage17_sparse_validated_roi_20261018/第十七阶段桨盘几何与ROI实验报告.html>)；[阶段结论](</D:/drone forecast/output/stage17_sparse_validated_roi_20261018/阶段结论.md>)。原始结果、冻结参数/源指纹与负结果均保存。

## 组成

- `frontend.py`：250µs/60–1600Hz活跃块FFT；周期支持验证已有椭圆/线段候选；复用阶段9状态恢复；三帧发布确认；50Hz局部ROI及≥10Hz全局重搜。
- `checks.py`：已知频率、奈奎斯特门槛、单极四盘、因果性、确认、空输入/失配检查。
- `study.py`：评分前冻结的四段固定RAW实验。**勿改冻结文件。** 最后控制台打印有已知int64序列化问题，analysis已先落盘；使用下面的包装入口。
- `rerun.py`：新目录重放及已完成目录续用；处理已知打印错误，按文件绝对位置加载本阶段报告，避免不同阶段同名模块冲突。
- `observe.py`：仅需RAW的当前观测入口，可选事件计数叠加视频。**不发布未来预测，不接RotorSensing，不输出机械转速。**
- `report.py`：只读推理输出后评分/绘图；`profile_fixed.py`：事后固定时刻热点诊断，非性能基准。

## 新RAW仅作检测

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\research\stage17\observe.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 5 --output 'D:\drone forecast\output\stage17_new_observations' --preview
```

输出目录必须不存在；`--policy global`关闭ROI，其余参数相同。默认roi。`--duration 0`处理至文件结束；实际处理可能慢于RAW时间，不建议先跑全长。最多100ms因果预热、纳入wall；不足预热时不发布。视频50fps来自RAW步长，不代表处理吞吐率。单份推理不加载标注或数据清单。

## 原固定实验重放

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\research\stage17\rerun.py' --output 'D:\drone forecast\output\stage17_independent_rerun'
```

新目录必须不存在。将冻结相同参数/源码，依次跑train6 22–27/108–120s、train4 aligned 74–84s、train5 0–8s。标签仅分析/展示读取；train4曾看过抽样，属于跨录像检查而非完全盲测。只有已完成case可`--resume`跳过；若发现未完成case，拒绝覆盖，选择新目录。已完成官方目录的resume路径已实际检验，原推理JSON不重算。

## 限制与下一步

原阶段16的四峰不是四桨真值；本轮椭圆/线段也不是独立电机轴真值。23/24/25s旧AI近似参考12点：20px内匹配2→12，但8px内2→0，说明空间分布改善与中心偏差同时存在，不能声称正式逐桨召回。

侧视5个漏标点中3个卡在连续确认，2个无整机提案；ROI在train4有精度退步。先保留原主轨迹，将周期证据作为评分/身份校验，分别消融发布迟滞和周期硬门控；性能先改事件直方图增量缓冲与真正裁剪几何。之后再接频率和预测，以同定位输入比较历史加速度基线。

未使用G；未访问或执行yolov10-main；旧src、旧默认与阶段9/16冻结源指纹未变。effqpdet/HelixTrack仅核查作者说明，未编译运行，不称作者完整复现。
