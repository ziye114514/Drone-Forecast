# 第十三阶段：谐波真实性与侧飞段降级

只读已冻结的 stage10/12 运行产物，不重跑推理、不解码 RAW、不改 `src/` 与阶段6–12。两个脚本：

- `harmonics.py`：同 tick 内的半频/倍频真实性 + 因果的“四桨转速可比”修正。
- `degrade.py`：rotor/body 两类发布模式的可观测性开关，及其对碎片与中心跳变的影响。

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage13\harmonics.py --run output\stage12_side_108_120_20261017
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage13\degrade.py  --run output\stage10_test_clear_side_clear_20261017 --method neighbor10
```

## 为什么要按“同 tick”重测

先前的“296 / 539 Hz 两族、比值 1.82”是按**身份在全段的中位数**得出的，而同一段里不同身份出现在不同时刻、对应不同物理桨，属于被时间结构污染的聚合。同 tick 重测后：清晰段 465 个多桨 tick 只有 16 个（3.4%）分族、离散中位 1.285；侧视段 703 个里 242 个（34.4%）分族、离散中位 1.233。谐波歧义真实存在，但不是侧飞段的主要矛盾。

修正规则只用当前 tick 的其它桨（因果），把偏离舰队中位数 ≥1.6 倍的测量折半或加倍；侧视段修正 162 次，离散 1.233→1.178。

## 降级策略的定义

- rotor 模式：本 tick ≥3 个实测桨且全部关联到已有身份（阶段11 的 `RigidTracker`）→ 位置/跨度/方向按实测发布。
- body 模式：其余 → 发布整机中心，跨度冻结在最后一次 rotor 模式值，方向声明为不可用。

在 108–120 s 窗口：rotor 模式只占 12.6–13.6% 的 tick；碎片 tick neighbor10 从 59 降到 3；**中心跳变（>50px）0–1 次出现在 rotor 模式、30–40 次出现在 body 模式**——模式标志因此是一个因果可用的可观测性开关。

## 边界

无真实姿态/转速/逐桨真值；碎片与跳变是几何定义。body 模式的“跨度冻结”是用陈旧但可信的值替换塌缩值，不是精度提升。策略不新增检出。
