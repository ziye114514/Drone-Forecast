# 检测模型与数据说明

2026-10-31阶段159新增**可选**小目标模型`models/small_target_native33_v1.onnx`，SHA256 `fc228a1b355a3b8396c05df63f7eb8beaf192bca758bf5089898e16e893a4255`，不是默认替换。继承此前FRED3/4/5三轮模型，再固定5轮匹配33ms原生窗口，FRED6验证、保留自采1/4弱标签；上游预训练可能覆盖全FRED，不能称盲测。空场5号60帧有6帧候选，仍有误报风险。详见[阶段159](RECALL90_MOTION_20261031.md)。模型保存在本机，尚未加入或上传公开源码ZIP。

## 当前兼容模型

当前检测后端固定输入 `[1,3,384,640]` FP32、输出 `[1,5,5040]`，单类无人机；原生帧尺寸1280×720。使用 Ultralytics LetterBox、置信度0.25、NMS IoU0.45，以及 ORT CUDA 图和固定IO绑定。

不能随意换成其他YOLO模型：形状、类别、输出布局和预处理都必须相容。仅更改路径和哈希不会自动完成适配。

`run_fast_structure.py` 当前实际默认权重为 `models/live_position_accumulated_v1.onnx`，SHA256：

```text
ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4
```

它是原stage40权重与固定10轮ROI增强微调权重的0.5等权参数平均，未增加推理网络。微调训练使用自采录像1/2/4、验证6；原定3号/空场5未参加该轮训练，但这些数据更早已用于开发。合并前ROI微调ONNX为 `326f23e3d93e1649325e57c04a9d0d1914265b39dd7630fc335145e59fca58c0`。详见 `DETECTOR_CANDIDATES_20261026.md` 的历史实验口径；其“默认仍原模型”标题描述的是当时的旧入口，不能代表当前精简RAW入口。

兼容旧入口 `configs/local.json` 仍指向原stage44 `best.onnx`，SHA256：

```text
e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b
```

本地参考来自stage40域适配、stage44导出的单类检测器；上游模型来源在历史协议中记录为 takara-ai/DroneStalker 固定提交 `93e846eef8abc974da1e0c5cf71331729648f143`。使用Ultralytics训练/导出接口。训练录像1/2/4、验证6、测试3；这些录像在更早开发中已暴露，不能称全项目盲测。

2026-10-31核查上游[DroneStalker README](https://github.com/takara-ai/DroneStalker#drone-detection)：作者说明初版YOLOv8n只用FRED0训练，之后YOLOv12 small使用整个FRED。尚未逐权重追溯其精确训练成员，不能因某段FRED未参加本地微调，就称检测器从未见过该段。本轮FRED8只作为预先固定、不按预测误差调参的过程留出，不是全系统独立盲测。上游README的性能宣称也不是本项目成绩。

源码包**不附带权重，也尚未提供经过整理的一键训练/下载流程**。因此当前是可审阅的源码候选包，未达到陌生用户clone后立即复现。下一步要补全可公开分发的模型获取/训练路径和相应上游说明。

## 数据

- 自采RAW用于开发演示，原始文件留在用户dataset，不打进源码包。
- FRED用于公共数据探索；不要把其下载数据整体搬进自己的MIT仓库。
- 人工框只用于事后评分，不用于当前推理输入。
- 公开视频片段是已有自采实验的ON/OFF结果演示；实际对外发布时由作者选择展示范围。

SDK独立安装，许可证不随本项目MIT变更。第三方代码和依赖说明见 `THIRD_PARTY_NOTICES.md`。
