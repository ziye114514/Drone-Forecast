# 阶段35：整机原生事件计数图LK补测

冻结body35/checks35/study35。只使用原生20ms ON+OFF计数，每10ms截止；复用OpenCV GFTT/PyrLK双向/相似RANSAC，非G重建，非HASTE/EKLT复现。几何成功播种特征，body阶段不补点；当前流支持不足立即停，不发伪造桨盘。短300ms/长2s均从独立几何时刻算。

输出`D:\drone forecast\output\stage35_body_feature_flow_20261018\第三十五阶段整机事件特征补测报告.html`。clear补20、side0、train3相似1/平移2，当前框覆盖无变化；train3未来均值改善来自新增一个目标，共同3目标完全不变。断流控制无当前发布，170ms后无新预测。普通1300 baseline六模型回归；真实1.9秒前缀用prefix35.py独立RAW核验。side仅108–113s，未包含全极端片段。

下一步若继续，应冻结新协议扩展全12s侧视，比较低点数平移/受约束补点并监测漂移。不要改35冻结源凑结果或直接推广默认。
