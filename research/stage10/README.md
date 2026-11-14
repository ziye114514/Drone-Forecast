# 第十阶段：自采连续段的整机身份与恢复审计

输出：`D:\drone forecast\output\stage10_continuous_identity_20261017`。先读 `阶段结论.md`。

本阶段回答一个具体问题：**在一段自采的连续录像里（含清晰、侧飞、离场与重新出现），整机模型会不会串桨、缩成碎片、重新捕获到错误目标。** 不改旧src、阶段6/7/8/9源码与结果；不使用重建G；不访问或修改`D:\yolov10-main`；不需要质量、尺寸、推力系数或同步飞控真值。

## 冻结内容（先写协议，再解码RAW）

`study.py`在解码前写出`protocol.json`，包含：区间、方法、逐tick处理方式（10 ms、冷启动、无隐藏预热）、失败定义、源码指纹。

- 三条自采录像、同一套方法与失败定义，构成一个批次（各自独立目录、各自冻结协议）：
  - **A 主实验**：`E:\dataset\train\6\media\event.raw` 第0–120秒全程连续，冷启动。含未出现、清晰四桨、反复入出画面、近距侧飞（110秒）、离场与再次出现（118秒）。
  - **B 第二个正样本**：`E:\dataset\train\4\aligned\event.raw` 第0–120秒全程连续，小目标、反复入出画面。
  - **C 负对照**：`E:\dataset\train\5\media\event.raw` 第0–60秒全程连续，标注为全程无目标；任何输出都是重新捕获到错误目标的直接证据。
- 方法：`strict10`（阶段8严格原生像素ON/OFF，作参考）、`neighbor10`（阶段8 3×3邻域，阶段9状态保护分支）。两者共享同一批tick。
- 失败定义（在`protocol.json`里冻结，任何跑完后都不改）：
  - **碎片**：已发布tick的整机跨度 < 0.5×当前有效尺度参考；无参考时 < 0.5×过去1秒已发布跨度的中位数。
  - **串桨**：相邻两个四实测候选tick之间，连续对应关系的相似变换残差超过 `max(8px, 0.06×桨间距)`，而某个替代排列低于它的0.5倍。
  - **错误重新捕获**：确认转移（`audit.transferred`）后的新中心，距上次确认观测外推中心超过 `max(80px, 1.3×整机跨度)`；或转移落在稀疏标注的absent区间内。
  - **覆盖**：同一批tick中已发布tick占比。

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage10\study.py `
  --raw 'E:\dataset\train\6\media\event.raw' `
  --output 'output\stage10_continuous_identity_20261017' --start 0 --duration 120
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage10\analyze.py `
  --run 'output\stage10_continuous_identity_20261017'
```

输出目录已存在会拒绝覆盖；`analyze.json`已存在也拒绝覆盖。新实验另建目录并重新冻结。

## 桨身份的现实

当前前端**没有跨帧持久的逐桨ID**：只要本帧凑齐四个候选，`constellation`就按本帧检出顺序整体重写，身份只通过“不足四桨时的部分布局传播”存活。因此本阶段审计的是“**能不能可靠维持身份**”这件事本身：

`analyze.py`对每对相邻四桨tick，在全部24个排列里找相似变换（旋转+等比缩放+平移）残差最小的对应。平面点集上的**奇排列无法由相似变换实现**，所以“最优对应是奇排列”就是一次真实的两个桨身份交换（串桨）；“刚性残差过大”说明这四点已经不是同一个物理四边形（碎片或被别的东西顶替）。

## 数据可观测性

`observability.py`对六条自采RAW做冷启动单窗可观测性抽样（每2秒一个10 ms窗），只统计候选与几何，不读标注，用来判断哪条录像在桨盘层面可用：

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage10\observability.py
```

输出在`output\self_capture_observability_20261017`。它只是单窗能力，不是跟踪结果，也不是召回率。

## 边界

- 画面位置仍是像素观测，不代表真实姿态、机械转速或身份保证；四候选不等于四真桨。
- 标注是320×184重建坐标的四字段稀疏框，只在事后核对用；本阶段推理不读标注。
- 一次离线运行的时间包含解码与JSON落盘，不是流式实时性证明。
