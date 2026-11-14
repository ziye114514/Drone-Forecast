# 阶段18：增量事件统计和轨迹确认

2026-10-18。四段35秒自采RAW，1750个20ms时刻。增量统计的严格分支与阶段17全局观测1750/1750完全一致；固定三个时刻的配对统计/频谱模块快2.52/4.84/2.71倍。完整清晰前端仍约67ms，未到20ms实时。

**不默认启用新轨迹发布。** 它将侧视发布307→316，但有机标签覆盖仍9/14；train4发布128→121。原默认入口保留。本轮只当前检测/定位，无未来轨迹提升声明。

报告：[第十八阶段增量统计与轨迹确认](</D:/drone forecast/output/stage18_incremental_track_20261018/第十八阶段增量统计与轨迹确认报告.html>)；[阶段结论](</D:/drone forecast/output/stage18_incremental_track_20261018/阶段结论.md>)。

## 文件

- `frontend18.py`：精确250µs分块直方图环形缓存；继承阶段17全局前端；共享提案的TrackPublisher（160ms保留、位置/尺度约束、无当前量测不发）。
- `checks18.py`：随机流频谱/完整几何等价，环绕过期、未来事件隔离、发布规则等检查。
- `study18.py`：固定35秒对照与配对模块计时。上述三文件已冻结，勿改后重跑或覆盖旧目录。
- `observe18.py`：只需RAW的独立研究入口，支持strict/track及MP4。**默认strict；track需显式开启，仅供实验。**
- `report18.py`：推理结束后读取稀疏框评分、可视化和尺度跳变诊断。它不参与推理。

## 运行等价增量检测

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\research\stage18\observe18.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 2 --publisher strict --output 'D:\drone forecast\output\incremental_strict_demo' --preview
```

输出目录必须不存在；start/duration须为20ms整数倍，duration=0为读到结束。最多100ms因果预热，计入wall；不加载标签、旧推理或数据清单。视频50fps按原始时间轴播放，不代表实际实时。输出只有当前观测，不接RotorSensing与预测器。

## 固定实验与复核

本轮固定输出目录在脚本中。四段已完成，不允许覆盖；`study18.py analyze`可只重算分析，`report18.py`只重做图与报告。新实验请建立新版本/输出并重新冻结，避免改动既有协议。所有RAW元数据/头指纹与阶段17一致；配置评分前冻结，未结果后调参。train4已看过抽样，只称跨录像检查。

清晰前缀15时刻两分支逐项一致；2秒RAW入口100时刻/97次发布，与正式track100观测一致。配对模块基准7次交替，包含新20ms入环成本，且频谱数组逐项一致。研究循环wall不是单分支实时基准。

2秒入口烟测后只将入口默认值改为strict，保留`cli_smoke/observe18_at_smoke.py`与原烟测源指纹匹配。最终默认入口另作100ms/5时刻检查；推理算法未改。

## 卡点与取舍

116.60/117.28s虽只60ms无发布且位置距离在限内，跨度比2.14/3.59导致重新确认；另3个侧视漏标时刻没有整机提案。继续延长TTL不对症。下一步用原neighbor10主轨迹、周期候选软评分和已确认布局匹配，避免局部片段更新整机尺度；性能再做稀疏缓存/连续内存布局与裁剪几何。独立密集框补齐后才评价未来轨迹。

复用既有SDK RawReader、阶段17/9/8几何/状态；查阅Norfair和Prophesee缓冲语义，但未执行Norfair或原effqpdet C++。无创新性或作者完整复现声明。未用G、未访问执行yolov10-main；旧src与默认保留。
