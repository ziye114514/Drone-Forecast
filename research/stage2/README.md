# 第二阶段：独立模块验证与跟踪回退

报告：`D:/drone forecast/output/stage2_module_validation_20261016/第二阶段实验报告.html`。

完成2160组解析三维圆投影、96组像素亮度阈值事件、公开周期目标探测、三条FRED训练序列前20秒，以及自有train6前30秒的组合验证。逐例CSV、协议、源快照、失败结果和真实未来对照视频均保留。

## 使用哪个入口

- 原`run_paper.py`保持冻结，原v0.3实验未改。
- `run_hybrid.py`是无约束活动回退的失败对照。它在部分FRED录像上提高位置覆盖率，但在自有录像出现大量背景误报与精度退化，不作为默认升级。
- `run_guarded.py`是后续开发建议使用的实验入口。保留原桨盘位置；只有近期四桨确认、位置连续性和中心偏移检查通过时才允许短时活动接续。未知活动不能自行确认无人机，不能提供姿态或推力。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_guarded.py' --raw 'E:\dataset\train\6\media\event.raw' --duration 30 --output 'D:\drone forecast\output\my_guarded_run'
```

输入仍只有RAW；配置和输出路径不包含真值标签。小目标无法看见桨盘时，身份约束会拒绝未知活动，不能声称同时拥有无约束活动方案的覆盖率。当前无真实实飞RPM/姿态独立真值；不声称恢复真实质量、推力系数、风速或实时性。

## 实验顺序

1. `study.py freeze`，再运行`geometry`、`raster`、`public_frequency`、`fred`。现有协议不可覆盖。
2. `fallback_experiment.py`冻结并运行初版；原始失败保留。
3. `fallback_fixed_experiment.py`记录重捕获逻辑修复，冻结后依次运行/评分三条序列；仅修复逻辑，没有测试集阈值调参。
4. `run_integration.py`记录无身份约束组合的失败，之后`run_guarded_experiment.py`冻结身份约束修复。后者属于开发修复，不是盲测胜出。
5. `build_report.py`、`finalize_report.py`生成评分和视频；后者只追加一次。
6. `validate_and_package.py`复核原代码指纹、1200帧真实RAW前缀一致性、物理门控、视频与9项新检查，并打包。使用原SDK对应的Python3.11。

主协议冻结脚本不能在原目录改后重跑；新研究使用新目录和版本。解析投影、像素事件模型和原算法有不同证据范围，具体限制见报告。
