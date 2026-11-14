# 观测错误机制研究与公开真值替代验证

研究脚本独立于已冻结的v0.3部署代码，不改既有成绩。

结果入口：`D:/drone forecast/output/stage1_observation_errors_20261016/第一阶段实验报告.html`。

## 已执行

- 360组主实验（20噪声种子×2初始增益条件×9观测场景）及18组当前状态真值诊断。
- 同一真实RAW共同历史、四条件事件删除与恢复观察。
- E盘现有FRED原生RAW/标注格式与对齐抽查；训练序列1前20秒冻结基线。
- 复用原EEPPR公开周期目标实验记录；不冒充本项目逐桨测速真值。

运行协议、源代码快照、逐种子结果、CSV、视频、替代验证边界均在上述输出目录。仿真预测误差、真实RAW干预差异、公开位置标注误差是三种不同的量。

## 复核现有结果

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\research\stage1\validate_and_package.py'
```

## 运行顺序

全新实验应改输出目录后依次：`run_study.py freeze`、`run_study.py controlled`、`analyze_controlled.py`、`raw_interventions.py`、`analyze_raw.py`、`build_report.py`、`prepare_public_substitute.py`、`public_baseline.py`、再次`build_report.py`。已有冻结目录不重跑freeze，不覆盖已完成的RAW运行。解码需已安装的Metavision SDK；路径和示例数据为本机路径。

部署仍使用原`run_paper.py --raw ... --output ...`。合成当前状态真值、干净学习器理想移植、过去标注CV均为诊断，不属于只输入RAW的部署程序。
