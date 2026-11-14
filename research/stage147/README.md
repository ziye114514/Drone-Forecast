# 147：凹曲率拆分粘连桨盘

2026-10-24。实现 `curvature147.py`，主入口 `--rotor-curvature` 自动启用146。
原144预测不变；新增路径可选、默认关闭。使用现有OpenCV、geometry._fit/_distance/_coverage/_uncertainty，外部论文只参考思路，没有复制源码或安装新依赖。

流程：当前阈值4轮廓→双尺度有符号凹曲率+凸包凹陷→凹点对→原始开弧的确定性稳健椭圆拟合→覆盖/残差/近似中心协方差→联合支持区域IoU与相容性→同源两子块成对分组，否则恢复父轮廓。没有将切割直线作为桨叶边界，也不强制四个分量。部分弧推断边界保持mode=arc/inferred_shape=True，单独允许合格弧中心建模，不能越过FOV检查；置信度降为原0.8倍。主阈值4/侧视阈值2/原146阈值8回退均保留。

最终四窗口1325实际RAW时刻在 *_paired；首版另1325时刻在 clear/side/train2/fred0，首版分析保留unpaired_analysis；原146对照保持。原main146快照run_optimized_before147.py，首版模块快照curvature147_unpaired.py。最终hash包含主入口与147模块。观测探针probe147/bench147读取当前ROI缓存或保存轮廓，不算新增端到端RAW。

|窗口|时刻|146→147四桨假设|成对拆分采用帧|0.4s基线→融合px|有效/缺失|处理中位|
|---|---:|---|---:|---|---|---|
| clear | 400 | 245→246 | 9 | 25.595→25.625 | 377/缺20 | 180.3ms |
| side | 475 | 57→63 | 13 | 50.027→49.959 | 66/缺9 | 135.3ms |
| train2 | 300 | 0→0 | 0 | 59.301→59.301 | 30/缺6 | 104.7ms |
| fred0 | 150 | 1→6 | 12 | 430.086→430.086 | 11/缺16 | 77.6ms |

四桨次数不是独立检测正确率。clear仅未来检测器自洽；side/train2人工框、FRED提供框均非COM/RPM/姿态真值。预测得分不支持将147升为默认；侧视物理分支仍逊于相同起点的144。FRED0仅按大目标选窗，不能外推整个数据集。

新增控制：9双椭圆接受6，4基础反例与80单椭圆反例无误拆；FOV/孤立子块回退控制通过。首版发现清晰35、侧视6、FRED2帧只取一个子块，最终每窗口均0。没有独立转速/姿态/轴心真值。

运行：`.venv-event311/Scripts/python.exe event_drone_forecast/run_optimized_forecast.py --raw "E:/dataset/train/6/media/event.raw" --output "D:/drone forecast/output/new147" --start 22 --duration 8 --rotor-curvature --preview`

复跑整个实验用batch147.py --suffix 一个新后缀（GPU串行）；分析用analyse147.py --suffix 相同后缀；controls147.py单独做几何控制。不要在GPU worker运行期间开另一套NN/大型分析进程。推理入口不读标签/旧输出。主预览视频只绘原144基线，新融合坐标看JSONL；报告图为同帧测量重叠。

来源：[凹点检测](https://arxiv.org/abs/2008.00997)、[作者项目](https://github.com/explainingAI/overlapped-objects)、[凹点轮廓分组](https://arxiv.org/abs/1906.01049)、[弧支撑椭圆项目](https://github.com/AlanLuSun/High-quality-ellipse-detection)。切分本身不声称创新。

接下来：结合短期共面结构，对同源原始弧分组稳定跟踪；优先解决重叠区域频率归属与谐波，不把活动组当电机ID。保留144默认，146/147实验开关分开。无运行中作业。
