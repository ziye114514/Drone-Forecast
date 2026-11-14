# 第十二阶段：把逐桨频率层接回新前端

输出：`D:\drone forecast\output\stage12_frequency_layer_20261017`（结论），运行在
`output\stage12_clear_22_27_20261017`（清晰段）与 `output\stage12_side_108_120_20261017`（侧视段）。

## 为什么要做这个

阶段8/9/10/11 这条线里**根本没有频率层**：`PaperPerception._frequency` 被有意写成返回全 NaN
（注释：Periodic measurements are owned by stable rotor IDs below），逐桨频率归 `RotorSensing` 所有，
而 `RotorSensing` 只在 `run_paper.py`/`run_hybrid.py`/`run_guarded.py`/`run_normals.py`/阶段1里被实例化。
结果是阶段8/9/10 的 `blade_pass_hz` 结构性为 null、`frequency_quality` 结构性为 0。
**“转速质量一直没过门槛”必须按“那个分支压根没算”来读，不能当成“算了不行”。**

本阶段把**未修改**的 `RotorSensing` + `periodic_measurement` 挂到**未修改**的阶段9 `RecoveryFrontend` 上，
逐 tick 记录每次逐桨测量的结果与拒绝理由。`src/`、阶段8/9/10 均未改动；协议在解码前写入，输出目录拒绝覆盖。

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage12\frequency.py `
  --raw 'E:\dataset\train\6\media\event.raw' --output 'output\stage12_clear_22_27_20261017' --start 22 --duration 5
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage12\analyze.py `
  --run 'output\stage12_clear_22_27_20261017'
```

## 结果（邻居10ms分支）

| |清晰 22–27 s|侧视/近距 108–120 s|
|---|---:|---:|
|处理tick|500|1199|
|检测tick|478|796|
|四桨频率全有效的tick|**449（89.8%）**|143（11.9%）|
|测量尝试|1914|3383|
|有效测量|1836（95.9%）|2171（64.2%）|
|有效测量 峰值/噪声 中位|939.1|354.6|
|有效测量 quality 中位|1.00|0.375|
|相位相干 中位|—|0.117|
|身份数（>1 s）|**4（4）**|40（14）|
|最长身份|>1 s，250–400 次有效|2.71 s / 250 次有效|

- 清晰段分秒有效性：22 s 0.75（预热）→ 23/24/25/26 s **1.00**；拒绝只剩 `warming_up` 10、`period_window_warming_up` 64、完成测量被门槛拒 4。
- 四个身份的中位桨叶通过频率：326.8 / 261.9 / 291.0 / 255.2 Hz。
- 侧视段拒绝分布：**完成测量但被 snr/quality 门槛拒 809**、`warming_up` 226、`period_window_warming_up` 63、`too_few_rotor_events` 62、`weak_periodicity` 52。也就是说侧视段的主要失败不是“事件不够”，而是**测完但质量不够**。
- 耗时：清晰段 5 s 素材 39.5 s（≈7.9×）、detect 中位 33.3 ms；频率层 sensing 中位 0.27 ms、P95 187 ms。

## 新瓶颈：半频/倍频歧义

侧视段 40 个身份里，有 ≥20 次有效测量的 26 个，其中位频率明显分成两族：
23 个 <400 Hz（中位 296 Hz）、3 个 ≥400 Hz（中位 539 Hz），比值 **1.82**。谐波歧义标志在多数身份上占 0.2–1.0。
这与阶段7“下一步要查半频/倍频”的判断一致，而且现在有了逐桨身份与相位相干，**可以按同一架机的四个桨互相一致性来解**：
四个桨的转速在正常飞行里不会差 2 倍。

## 边界

`blade_pass_hz` 是桨叶通过频率代理；`blade_count=2` 是假设，机械 RPM 仍需独立叶片数与谐波验证。
侧视段的“有效”只表示过了 snr/quality 门槛，不代表真值正确；没有实飞转速真值。身份数为 40 说明该段身份抖动仍在
（与阶段11 的关联结论一致）。所有数字来自两条自采窗口，属开发片段。
