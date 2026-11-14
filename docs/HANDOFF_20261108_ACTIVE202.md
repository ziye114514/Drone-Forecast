# 2026-11-08 进行中交接：192–202，非最终收尾

## 后续203–211进展（优先于下文旧“尚未”状态）

- 203 public plane-adaptive机制测试通过：wrapper与202离线0px、未来后缀不能改变前缀、目标成熟前不使用、协方差PSD、freeze保留混合。新录像12先冻结后运行，181帧4个区域、0有效预测，不能评分；12.0s原始ON/OFF明确有清晰大无人机但逐像素period=0。新12已成为开发数据，不能再称盲测。
- 204–206原生椭圆种子/距离算子：65真实弧20接受完全一致，206成对计时132→97ms；算子计时非RAWFPS。210已接公共`--rotor-native-geometry-dll`显式选择，Python默认保留。录像13共享GPU全181帧原生/Python adaptive两臂body/geometry/structure/coefficient/force/physical/valid差异全0，176有效。脚本末尾数组dict比较抛错，结果此前已完整保存；独立post-run比较和source哈希确认完成，completion明确记录此事。
- **208发现193/194原型存在统计伪周期，旧区域恢复计数不能称收益**：每1ms首事件投票会把均匀噪声变成约960Hz。已改为与时间无关的SplitMix64最小hash抽样，短block均值+线性趋势去除、方差取最坏界。8组稠密均匀负例0通过；稀疏240Hz正例通过；NumPy与原生C++直方图输出严格相同。频率是区域候选，非电机RPM/推力。主预测/201/202未启用旧恢复，不受影响。
- 209修正后原型在12：区域4→28、位置观测3→17、有效0→14，24包恢复；不再称194的旧结果可靠。恢复平均83ms，中位76/P95144，不实时；没有新FDE参考，不能声称精度下降。已给regional-only恢复清空逐像素频率/相位输入，隐藏推力全部进入未知力，不用区域周期冒充电机RPM。
- 当前真正精度增量还是202成熟误差信用：15/14/16均值改善3.92/1.62/3.34px；13退.35、side退.008、train2退.373，未统一达30px，默认176未替换。
- 210完整RAW13 adaptive+native geometry正在跑；勿与原201不同模型测速直接当纯算子收益。211 CUDA Graph区域统计原型在开发，初次GPU解释器SDK导入/driver上下文错误已纠正，尚未通过测试/接公共入口。GPU RTX5060Laptop7.93GiB。不要把原型计时称RAWFPS。
- 无新子代理/安装/上传/公开包改动。最终checkpoint/HTML尚未整理，主checkpoint暂191。RAW210运行期间不要编辑SOURCE_FILES列的文件。

用户最新：精度该怎么受益，想清楚往那里攻；去休息，持续做至额度耗尽。不按录像打补丁。两处yolov10/RAW/权重/公开NN包都不改，禁止新子代理，用户没同步日志不用再问。当前无后台作业，工作未结束。

## 已完成与真实数值

- 192原生事件损失审计：ON/OFF总比例约平衡，不是简单整体单极性。弱片段大量像素33ms内只出现一两次，强度/重复门槛后很少；固定小热点会产生240Hz，不能把热点周期当无人机。
- 193/194区域无符号事件时间检验：16px格、早半窗发现/后半窗确认、65us bins、1ms条件包络、同像素1ms只投一票、至少12不同像素，防热点。是候选区域周期，非单像素/电机RPM；原创实现借鉴EEPPR区域重复/DDHF梳状谱，未复制作者代码。CPU较重，未接公共实时入口。16区域152→166/181，有效142→158；14区域139→141/有效124→125；17区域7→37/有效0→32；19区域0→20/有效0不变。16新增都在15.033–15.533接近出画，14新增14.033/15.4；已有固定.4s计分点不受益，不能冒充精度提升。新检测尚未逐帧真值核验，不是AP/召回。
- 195动作用已有20ms native counts替代33ms，period/geometry不改：15均值52.867→52.208但14/16均退，默认False。196 ON/OFF独立匹配分歧加R：15几乎不变52.881，分歧中位.087px，不默认。198在线阻尼λ先验物理响应使15退到62.836；199相同总Q沿normal分给h使15退到54.607，均不默认，不再扫这些标签参数。
- 200/201 SeparatedForceObserver：a=k*u+d，d=e+h*n；h快变沿normal，e有平滑OU变化率j；j用于未来外推。14state，初始h/j均值0，既有1000/.35先验，无NN/外部标签/机型参数。当前法线观测只在过去区间之后更新；known/unknown transfer保净力。public `plane-separated`已接两入口，默认free/176保留。
- 同源.4s 191→separated：15 52.8673→44.9421(25)，14 44.7935→40.3874(15)，16 48.0089→42.8117(22)。13预先缓存/窗口声明且算法冻结后才看结果：37.1334→37.0619(25)，P95 92.0264→96.3386变差，不能称普适泛化。13/14/15/16是弱事件轮廓参考counts8/4，全部27帧已逐页目视检查，不是独立人工GT/COM。13/14有出画/支架遗漏，完整参考和失败保留。
- 同资格、同fast-noise trace的各向同性控制：13/14/15/16均值37.5361/41.8486/50.9614/45.6976，说明方向分工有作用，收益不是仅加噪声/状态。no_rate同状态控制分别37.2426/42.2032/45.1374/44.2179；rate额外作用有限但部分正。最初GPU201 isotropic不匹配无normal时资格，正式用独立 `MatchedIsotropicControl` 离线同输入更正，主模型没变。
- 旧人工标签cross-check：side191 42.0662→separated42.2243略退；train2 86.7936→98.6402明显退，不能全局替换。旧side评分仍18/19，精确左锚点修复后的口径，不能拿旧17点38.62混比。
- 完整RAW201/13已实测：steady15.4801FPS、P95周期184.01ms、无丢包181；实际有效发布24/181，仅2.064Hz，source age median3284.65ms/P955260.76ms，未持续30Hz。CPU geometry mean42.8/P95166.77ms，motion4.62/7.05ms，support3.02/6.14ms；GPU service30.09/39.37ms。主要瓶颈几何，不是14state小滤波。RAW启动13.59s、总33.05s。100us cache/RAW源格差导致少量门槛差，模式/输入整体一致，未发现入口错误。全部保留，不称30FPS。

## 当前正在接入：202因果增量信用

复用已有 AdaptiveForceMemory 的目标成熟、reference rebase、3s历史/.35衰减/6px Huber/36 ridge/每.2s原点/三时域除3规范；只把pending delta换成 separated预测−191预测。未成熟目标不读，当前body实际位置作自监督，不读labels。初始weight=0，当前plane不合格不启用。避免用录像ID选模型。不是IMM；没有实现FilterPy IMM，只读了原始文档。

离线202同起点均值：15=48.9461，14=43.1759，16=44.6710，13=37.4869，side=42.0741，train2=87.1664。相对191三段改善约1.6–3.9px，13轻退.35、side+.008、train2+.373，基本挡住旧train2直接替换+12px退化。不能宣称全场景无退化。物理增量只在成熟误差支持时加权，全部无NN。

`causal_response_credit.py`和`AdaptiveResponseObserver`刚接公共 `plane-adaptive`，尚需接口/离线一致性/因果检查与新的录像12冻结测试。API所有旧mode保留。publisher支持force.response_model_mixture，freeze会深复制。primary内部x/P/seen用于reference/entry bookkeeping，不是混合后后验；输出coefficient、force均值与sigma显式混合，nested components决定精确连续h外推。aux two_point_init同步，长gap primary reset会重建aux+credit。注意检查wrapper与离线helper202逐帧相等和数学分区元数据。

## 下一步按序继续

1. 完成plane-adaptive机制/固定输入parity（尚未做），13/14等已经用过，不再当它的盲测。
2. 预声明录像12的10–16s，在修改完成后冻结源，再解码/运行/看参考。缓存12还没建立。不要根据12结果改参数。
3. 速度攻击可考虑 exact arc AMS initial hypotheses batch/parallel，刚读geometry171：65真实弧20接受可复用 `tools/check_batched_arcs171.py` 的geometry170 NPZ；当前65弧串行约55ms。`fit_arc_exact`对18个seed逐个Python cv2.fitEllipseAMS再批量残差。MSVC18 Community现成，SDK4 opencv libs现成但Python OpenCV5；若写native batch假设算子，必须实测接受/数值一致，不默认假定版本一样。尚未写/编译任何新AMS代码。也可优先只做精度交付，不把算法质量换掉追FPS。
4. 最后写HTML曲线/预测图、WORKLOG正式结果、最终交接和checkpoint。目前checkpoint和README还停191；本文件是进行中补充。所有结果目录含source_snapshot，修改了公共代码所以旧freeze SHA不能当当前tree仍同一版，真实运行快照保留不动。

关键目录：output/timing194_20261107、motion195_20261107、polarity196_20261107、damping198_20261107、normal199_20261107、separated200_20261107、separated201_20261107（scores_all/scores_old_labels/checks/raw13）、credit202_20261108（scores/active_progress）。所有已知shell sessions已完成，无待等GPU作业。

文献 primary：EEPPR https://arxiv.org/html/2408.06899v3 / https://github.com/JackPieCZ/EEPPR ；DDHF https://arxiv.org/abs/2603.08386 ；rotor drag https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf （只作响应建模启发，未实现完整3D drag）；FilterPy IMM https://filterpy.readthedocs.io/en/latest/kalman/IMMEstimator.html （未集成，不把增量混合称IMM）。不复制未许可作者二进制进公开包。无新依赖/训练/上传/子代理，目标30px/持续30RAWFPS未完成。
