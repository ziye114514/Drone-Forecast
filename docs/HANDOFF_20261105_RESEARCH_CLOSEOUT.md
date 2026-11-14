# 项目收尾：研究快照发布准备

2026-11-05，用户希望及时止损，询问当前项目能否作为GitHub开源发布及如何说明。**本轮停止主动算法优化，不再自行启动RAW/GPU负载、训练或预测参数实验。** 若以后要恢复研究，应由用户重新提出。30px/持续30FPS与物理增益目标仍未完成，不标为研究成功。

## 当前交付

`D:/drone forecast/output/github_research_closeout_20261105`：

- `event-drone-forecast`：标准源码目录，131文件；整体是research-snapshot-20261105。
- `event-drone-forecast-research-snapshot-20261105.zip`：约6.6MB，附SHA256；放GitHub Releases用。
- `ROOT_README_FOR_NESTED.md`：适配用户既有“外部README＋event-drone-forecast子目录”的完整README源码。
- `root_files_for_nested_layout`：外部README、LICENSE、NOTICE，避免根目录许可缺失。
- `GitHub项目简介与发布文字.txt`：仓库简介、Release文字、中文项目介绍及简历用语。
- `网页上传第1批`、`网页上传第2批`：90和41个源码文件，保留嵌套布局，符合当前网页一次100文件上限。
- `发布操作说明.txt`、`delivery.json`及`verification/checks.json`：交付和检查范围。

没有操作GitHub远端、创建Release、删除旧仓库或归档线上仓库。用户自行选择是否上传，不需要继续做算法实验作为发布前提。

## 代码组成与真实性

1. 从已经整理/验证的2026-10-28 v0.4.0-alpha.2公开候选目录复制旧RAW基线，先核对原SOURCE_MANIFEST；除README、NOTICE、来源/文件清单、CI轻量任务外，旧运行代码不变。不是导出整个本机工作区或声称最新RAW全链已达标。
2. 当前阶段169的RotorEvidence/VisibilityLayout与FiveCircleModel放入独立命名空间`src/event_drone_forecast/rotor_observation`；只改一个相对导入，几何数学与本机源码逐字/哈希校验。没有自动替换进旧RAW实时链。
3. 新`rotor_demo.py`和120行小型数值候选，让无SDK/GPU用户能复现后端接口。该候选源于自采录像已提取的弧/椭圆/周期，上游用了NN辅助ROI/位置；候选不是真值，不是纯桨叶AP/RPM/轨迹精度证明。
4. 包内README明确64个删点案例仅来自16个相关帧，相对完整检测布局的一致性0.837→0.325px不是真实遮挡或0.4s FDE；周期响应标定—后段检验无稳定增益，负结果保留。细节仍见阶段169交接，未改其原输出/快照。
5. 新快照沿用原公开集成版AGPL-3.0-only，原MIT与第三方许可保留；未覆盖研究目录MIT文本，未对第三方数据/权重/SDK作新授权。排除RAW、FRED原始数据、检测权重、SDK、论文PDF、二进制及未授权研究仓库。保留AI辅助实现说明，不将第三方方法称作原创。

## 检查与限制

78个Python文件语法解析；34个Markdown相对链接有效；未发现检查规则覆盖的私人工作区路径；旧运行源码未改；没有不应分发的RAW/权重/SDK二进制。ZIP CRC和清单核对后，解压到独立目录、Python -I模式运行新机制检查、120行示例和旧CLI帮助，均通过。示例输出20完整、4近期三点、14二点、77侧线、5未解状态，与现有后端一致。

第一次发布检查发现旧README搬到docs后链接基准变化及新README一个快速开始链接名不对，已经修复再封包；这只是发布路径修复，没有追加预测实验。干净机器完整RAW部署、性能、真实遮挡/预测精度没有新增证明。本轮无后台推理任务。

## 发布定位

“事件相机桨叶观测、可见性条件化几何恢复与二维短时预测研究原型。”公开工程实现、示例和局限，不写首创/SOTA、可靠真实推力恢复或30px/30FPS达标。维护状态写为冻结研究快照、暂无持续更新承诺。

工作区根WORKLOG和latest_handoff已更新到本收尾文档。下一步仅由用户决定是否上传本地准备好的快照；不自动继续优化。
