# 2026-10-26模型候选：默认仍保留原模型

这轮完成了固定10轮局部放大增强训练（355+199张训练、180+93张验证，原定测试3/空场5不参与），以及固定0.5参数平均。标准方法复用，不是新网络。全部权重仅在本地，未公开上传。

|模型|ONNX SHA256|状态|
|---|---|---|
|原模型|e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b|默认保留|
|放大增强微调|326f23e3d93e1649325e57c04a9d0d1914265b39dd7630fc335145e59fca58c0|检测迁移改善；短时预测门槛未通过|
|固定等权参数合并|ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4|可选候选；原两段小收益、留出运动均值略差|
|混合FP16|6928f32ce22f7c8d9cf710eba4d06091521231390690b24a08a2e5b0f070a201|仅19图数值/前向探针，未跑完整RAW，未默认|

单模型+尺度实际RAW：侧视固定0.4s n16，同目标原模型离线参考52.583→51.368px；train2 n6 88.629→85.181px。P95略差，不作为普遍升级证据。原定留出3号0—8s的13个变时域(.317—.451s)目标，60.994→61.770px；不可与固定0.4s跨口径比较。

两个模型同时推理的等权中心结合保留为可选诊断，命令参数`--center-ensemble-weights`和`--center-ensemble-sha256`。保留原检测资格、框尺寸和测量噪声，第二模型不能初始化或延长缺框。额外推理有真实耗时，不能只引用冻结时刻的较好分数。

原生周期观测的`--periodic-sampling fixed-quarter`与`--periodic-packet latest-pixel`仍为可选。后者只适合无状态141观测，不能作为状态化频率/相位引擎输入。都没有变成推力测量。

从项目根目录运行候选（需本机现有SDK/GPU环境；这是源码命令，不是新wheel）：

```powershell
.\.venv-event311\Scripts\python.exe event_drone_forecast\forecast.py run `
  --config event_drone_forecast\configs\local.json --profile live `
  --raw E:\dataset\train\6\media\event.raw --start 110 --duration 4 `
  --output output\my_single_model_candidate `
  --detector-weights output\equal_weight_detector_20261026\merged.onnx `
  --detector-sha256 ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4 `
  --frame-transport frames --startup-sync reader-ready --scale-rollout compatible --preview
```

以上不会改变默认配置。`--dry-run`可只核验参数，不启动推理。命令行模型路径相对当前目录；配置文件内路径仍相对配置文件。

密集标签评分：`python tools/score_live_labels.py --run-dir <完成的live目录> --labels <GUI生成labels.json> --output <新分数.json>`。按实际发布时间+h，不用旧stage15评分器冒充live时间语义。留出3号80帧在`output/dense_labels_train3_transfer_20261026`，尚无新标签，没有自动启动GUI。

最后实际源快照在各RAW目录。其后仅增worker精度说明、CLI模型覆盖和评分工具，未声称再次完成整个RAW验证。旧42文件源码ZIP和0.3.1位置wheel均不含本轮全部改动，不能沿用旧包名宣称已更新。
