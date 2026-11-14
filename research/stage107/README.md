# 107：中心运动与外观偏移分离

入口 `run_center_bias_forecast.py`。保留104全部60路，添加body-only KF、100ms校正平滑、固定噪声偏移KF和视角噪声偏移KF四条候选，默认54不变，API仍93。

状态 `[cx,cy,vx,vy,bx,by]`，当前框中心为`c+b`，合格四桨几何约束`c`。首次几何对齐后的图像坐标偏置在同一对象期间保持，104几何过期不直接重置运动状态。物体超过500ms没有body观测仍按原身份超时重新初始化。协方差和框/几何相关项是假设，未经概率校准。

直接执行固定FilterPy提交3b51149的原predict/update/reshape_z函数AST，无数值改写，MIT许可证/完整源和SHA保留在external/filterpy_linear_upstream。不安装filterpy。增加的是状态和观测设计，不是新的卡尔曼基础算法。

实际运行：

```powershell
.\.venv-event311\Scripts\python.exe event_drone_forecast/run_center_bias_forecast.py --raw E:/dataset/train/6/media/event.raw --output "D:/drone forecast/output/my_center_bias_trial" --start 22 --duration 5 --preview --verify-frames
```

目前固定噪声版清晰段0.4s稀疏框中心误差18.537px（普通KF22.863、简单平滑22.236、104中心KF19.106、原融合13.708）。侧视没有模型初始化，固定版等同bodyKF81.723，视角降权反而94.086；不能据此宣称解决侧视或达到实时。

106固定110—119.5s连续RAW没有四桨初始化。108复用当前20ms候选增至14个合格拟合，最长连续3帧，仍不放宽5帧门槛。用户核验C112.02s四盘可见，而10/20ms均只3候选，是下一轮提取链路诊断的重点；A仅3盘，不强行补四。

本轮真实RAW815父/1630原生（含独立前缀和无机）、原60精确、新4精确缓存；相同算法维度的前缀严格相等。4D/6D无几何等价控制出现4.55e-13浮点差异，单独校验器用1e-8容差，原冻结推理源码不改。该适配只涉及验证。

受控验证请运行 `python event_drone_forecast/research/stage107/synthetic_checked107.py`；原 `study107.py synthetic` 保留初次过严的跨维度逐位断言，故未作为最终验证入口。缓存、RAW、评分仍使用study107各子命令。

结果、全部负控制、原生核验图、用户反馈及视频见output/stage107_center_bias_filter_20261022。下一步先改可见第四盘的提取/合并问题，分别评估姿态与位置可靠性，再做连续恢复实验。无新RPM/风/质量真值。
