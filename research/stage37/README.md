# 阶段37：公开整机YOLO迁移

固定DroneStalker93e846eef8abc974da1e0c5cf71331729648f143的best.pt，Git blob/SHA验证。Py313已装Torch/Ultralytics安全读取EMA并导出纯张量，严格网络重建；原EMA和新网络零/非零/真实图原输出完全一致。未执行上游应用或训练，无G/yolov10-main访问。

结果`D:\drone forecast\output\stage37_upstream_detector_20261018\第三十七阶段公开整机检测器迁移报告.html`：40时刻、3表示、10/50ms，共240图，conf0.25全零。9张FRED原事件帧功能对照7出框，训练重合，不能当FRED无污染基准。暂不接默认预测，下一步固定权重做背景过滤/尺度对照。

执行顺序：study37.py freeze → prepare_forward37.py(SDK311) → predict37.py(Py313) → score37.py → probe/control辅助及report37.py。所有核心已冻结勿改；首次v1真值元数据已在推理前拆分，保留freeze_v1_before_inference；原study37.prepare不支持SDK倒退定位，应使用冻结补充的prepare_forward37.py，失败部分图保留。推理访问GT文件会拦截。源/模型/输入/输出指纹见协议与preparation/inference记录。
