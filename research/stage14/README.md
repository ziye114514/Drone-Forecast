# 第十四阶段：二维运动模型对照

把 Target_KF 的思路（CTRV + IMM）在**同一份定位输入**下和现有基线比。只读已冻结运行产物，不重跑推理、不解码 RAW。

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage14\motion.py `
  --run output\stage10_continuous_identity_20261017 --method strict10 --horizons 0.1,0.2,0.4
```

五个因果递归滤波器：`cv`（匀速KF）、`ca`（匀加速KF）、`ctrv`（二维恒定转率+速度EKF，Target_KF `EKF_CTRV` 去 z）、`imm`（CV+CTRV，结构与 Target_KF `IMM` 同）、`cv_hist_acc`（阶段7 强基线：CV+过去 0.35 s 拟合加速度）。

**参考值是同一检测器在 t+h 的输出，不是独立真值**——衡量外推质量，不是绝对精度。`--horizons` 可换，用来检查结论是否随 horizon 变化（Target_KF 用的是 0.5–2.0 s）。

结果与结论见 `output\stage14_motion_models_20261017\阶段结论.md`：0.1–0.4 s 上 `cv_hist_acc` 最好、CTRV/IMM 无收益；1.0 s 上 IMM 反超，而 `cv_hist_acc` 因过冲变差。

## 边界

无真实位置/姿态/转速真值；参数在评分前固定、未逐窗口调参；本轮未接频率项。
