# NN主体GitHub发布收尾

用户最新要求：以较强NN主体版本发布，五圆/可见桨数/周期二力思路作辅助，注明辅助作用不大，属于一次失败的创新尝试。本轮只整理发布和复原检查，没有恢复算法优化。

## 唯一推荐发布目录

`D:/drone forecast/output/github_nn_main_20261105/event-drone-forecast`。

整体版本`v0.5.0-research.1`（Python元数据`0.5.0a1`）。ZIP同目录外层`event-drone-forecast-v0.5.0-research.1-NN-main.zip`，约7.34MB，214文件；外部README全文`ROOT_README_FOR_NESTED.md`，根许可文件`root_files_for_nested_layout`，网页上传按90/90/34分三批，发布文字见`GitHub简介与发布说明.txt`，操作见`上传说明.txt`。旧几何主体包和旧ZIP保留，不再作为本次推荐。

## 发布内容与口径

- 默认主入口`nn_forecast.py`：NN负责检测定位，后续事件图像KLT、状态估计与模型外推；没有端到端神经轨迹网络。
- RAW主体选保存的2026-10-31清晰段NN运行快照，公共路径/导入/配置/源码指纹适配。不是全历史所有入口在统一盲测上证明的全局最优版本，文档使用“较强已记录NN主干”。三个片段原NN调度/恢复配置略有差异，不拼接成新包统一RAW成绩。
- NN原基线本身已有旧几何/周期线索，不能将其称为彻底剥离所有桨叶物理的纯方框法；新增166辅助作为额外增量检验。
- 新辅助置于独立命名空间`event_drone_forecast.experimental_nn_aux`，不覆盖主干依赖；默认关闭。支持`nn_forecast.py demo --rotor-assist`或`nn_assist_replay.py --mode rotor-assist/matched-zero`固定输入因果回放；**尚未串进发布版实时RAW链**。
- 阶段169几何接口继续作为独立附加模块保留，不拿0.325px布局一致性当轨迹精度。纯桨叶源码工作区不删除、不改成NN成绩。

历史源时刻0.4s均值：清晰6/6点27.555→26.889px，同资格置零26.854；侧视17/19点37.110→37.653，置零37.671；另一录像7/7点68.193不变。清晰约+.666px收益，侧视-.543px，另一段0。未证明周期数值净收益；公开表述“辅助作用很小且不稳定，本次创新尝试未验证成功”，不推论所有物理方向被数学否定。

## 复核

`verification_2/checks.json`记录最终检查。第一次隔离检查暴露新CLI在Python -I模式下没有加入自身模块目录，已修复发布入口并另存第二次检查，未删除失败记录。

独立解压、Python -I运行NN帮助、RAW dry-run、辅助机制检查及三片段baseline/aux/matched-zero九组复原。NN位置/速度/有效性保持；三片段120/132/141行所有预测相对阶段166已落盘结果最大差均0px。再阻断Metavision/Torch/Ultralytics/ONNX Runtime导入，辅助demo通过，证明回放入口不依赖运行NN/SDK。没有新RAW/GPU推理、训练、模型下载、精度搜索或FPS测试；不是干净机器完整RAW部署证明。

RAW实际运行仍需合法SDK、CUDA相关环境、兼容模型；此冻结NN配置默认JIT，需requirements-counts.txt中的Numba/llvmlite。SDK二进制/权重/原始RAW不入包。根AGPL、原MIT、FilterPy原许可及AI辅助说明保留。

本地README和ZIP已准备，不曾创建GitHub远端提交/Release、删除线上仓库或改变可见性。用户依其既有手动上传流程发布；若未来希望代上传，需明确仓库路径。项目仍冻结，不自动重启优化；30px/持续30FPS未达成，研究目标不标为成功。
