# 第三阶段：多桨法向、残弧置信度与预测检验

报告：`D:/drone forecast/output/stage3_normal_confidence_20261016/第三阶段实验报告.html`。

本轮实现已有方法作强对照，不作创新性声明。1620组单圆/残弧、2160组多圆/内参扰动、10种子动态机制测试，以及三个真实RAW片段共6600帧。

## 主要结论

- 已知内参的完整四圆投影中，联合候选能减少法向初始化错误；不等于真实RAW的姿态精度。
- 局部敏感性与连续块删除仍会接受稳定但错误的短残弧，不能视为已校准置信度。
- 实际录像没有相机标定，只验证弱透视共识/门控对二维预测的影响。0.4秒融合误差在train6略好，另两段略差；未选默认升级。
- 三臂共用位置/转速，CV状态逐帧相同；新分支更换完整轴向滤波/准入实现，包括噪声与差分代理处理，不是单因素候选选择消融。

## 运行研究入口

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_normals.py' --raw 'E:\dataset\train\6\media\event.raw' --duration 30 --output 'D:\drone forecast\output\my_normal_ablation'
```

只输入RAW，输出frozen、consensus、consensus_audit三臂。无标定时不会造一个K来跑真实透视解。原`run_paper.py`和`run_guarded.py`保持原样，研究入口不替代默认预测器。

## 实验与复核

1. `experiments.py freeze`后运行`single`、`multi`、`temporal`；冻结协议不可覆盖。
2. `analyze.py modules`检查候选选解、接受率、条件错误率与控制预测结果。
3. `raw_experiment.py freeze`后运行`train6`、`side6`、`train2`。`raw/`保留首次接口失败，`raw_v2/`为修复报告字段后另行冻结的运行，未调几何公式或阈值。
4. `analyze.py raw`事后读取原有人工框评分；`render_videos.py train6/side6`仅读取未来RAW用于展示。
5. `build_report.py`生成报告；`validate_and_package.py`复核源码、共享CV、2600帧观测日志前缀回放、9项新检查和视频。

观测前缀回放不是新一轮原生SDK前缀解码；合成动态测试使用当前真实p/v隔离姿态作用，不是部署成绩。全部证据范围见报告。
