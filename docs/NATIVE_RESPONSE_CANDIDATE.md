# 原生桨叶未知力响应候选

## 2026-11-10 当前版本（以下旧74教师说明为历史）

默认模型已改为`models/native_response_plane_v1.json`：一个共享模型，186个原生弱教师来自7段其他录像，评测3/4/11/13/15/16整段均未参与拟合。同102固定起点、0.4秒均值**43.3167→36.4630px**，P95 **102.268→75.743px**，35px内49→59、30px内45→51。六段分别**27.893/33.413/38.177/30.371/58.285/59.969px**。35px整体目标仍未达到；这6段都已暴露开发，弱轮廓参考不是真实质心/运动捕捉真值。

入口仍为下面的`run_rotor_learned_candidate.py`命令。可加`--response-model 'D:\drone forecast\event_drone_forecast\models\native_response_visibility_v1.json'`恢复402候选（36.919px，35px内63个）。原纯机械入口与公开NN包保留。36.463均值更低，但35px内起点数比402少4个，P95也略高，不能称全面更稳。

新模型含12个残差系数和1个未来法向驱动增益：保留原桨盘物理链，增加当前/过去的可见桨数与扁率上下文，以及当前无向法线投影下的运动响应。已知可见输入、隐藏力归属与当前p/v/a不变，附加项是未来未知净力变化率的统计先验。驱动增益不是牛顿推力标定；同系数把该增益固定1反而36.336px，因此**推力幅值标定的独立收益尚未证实**。当前法线条件项的匹配消融为36.710→36.463px，支持小幅几何信息收益。

实际GPU工厂`output/factory413_20261110`复现13号25点27.893px，181包预测和基线当前状态差均0。`output/controls416_20261110`：固定机体、对称桨盘摆动及4→2→4变化没有创造运动（0px）；法线翻转差0；实际发布年龄求值差6.9e−13px。它们不是完整RAW持续30FPS认证。

扩大训练读取了其他13段RAW的固定40–70秒（短文件自然结束），223个新教师与旧186共409个。411/412为38.738/38.667px，未替换默认。414只用训练源留出选择权重，选中原方案，仍38.738px。415二次阻力变化项训练留出略好，但六段38.981px，未采用，不继续扫参数。所有结果保留；没有按录像选版本。

复现当前13参数模型用`tools/evaluate_combined410.py`（输入完整406/409冻结教师）；原始训练输入由`tools/capture_training400.py`生成。新录像必须先按完整录像划分训练/测试；当前旧`tools/train_native_response.py`只训练5参数版本，不会生成本模型的12参数形式。不要把旧训练示例误当当前模型的完整复现。

新入口`run_rotor_learned_candidate.py`继承原生ON/OFF、周期种子、可见桨数/侧线分支、五圆约束和当前可见项＋未知项。只新增未来图像域未知净力变化率的条件先验；没有机体NN、G通道、隐藏转速补全或牛顿推力真值。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' `
  'D:\drone forecast\event_drone_forecast\run_rotor_learned_candidate.py' `
  --raw 'E:\dataset\experiment\15\media\event.raw' `
  --start 10 --duration 6 `
  --output 'D:\drone forecast\output\native_response_new_run' `
  --gpu-python 'C:\Users\子夜\AppData\Local\Programs\Python\Python313\python.exe' `
  --rotor-native-geometry-dll 'D:\drone forecast\output\uncertainty216_20261108\geometry_kernels.dll'
```

输出目录须为新的目录。以上是运行示例，并非这条RAW命令已经取得30FPS的证明。实际GPU工厂`output/factory399_20261110`复现了15号固定坐标残差交叉训练模型34.298px，预测坐标差2.7e−12px；它没有完整RAW解码测速。旧桨盘坐标残差模型的392成绩36.002px亦保留。

当前默认模型`models/native_response_fixed_frame_v1.json`在6段已暴露开发录像的74个原生教师上训练，供下一段新录像测试。38.620px是按整段录像留一、六个不同训练折的同102起点研究成绩，**不能说这个共享模型在其训练录像上独立达到了38.620px**，也没有35px/30px精度保证。旧桨盘分解模型`models/native_response_v1.json`与原纯机械入口、已发布NN版均保留。

默认模型的5个响应系数使用固定图像坐标中的运动和制动状态；原有已知/估计的桨盘轴向响应仍随法线演化，新增未知净力残差不随桨盘旋转。模型不接收录像ID、位置坐标、帧号或未来控制。可见桨叶的当前输入与隐藏项归属保持，发布时用同一未知净力增量在实际输出年龄＋0.4秒求值，未用旧预测线性插值冒充新预测。它是统计控制/扰动先验，不证明单目观测已唯一分离推力、风、重力和阻力。

模型只从已完成推断的原生当前/未来轮廓位移提取教师，不读人工评测框。完整数据源须先按录像划分：

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' `
  'D:\drone forecast\event_drone_forecast\tools\train_native_response.py' `
  --predictions 'D:\path\training_recording_a.jsonl' 'D:\path\training_recording_b.jsonl' `
  --residual-frame camera `
  --output 'D:\drone forecast\output\new_response_model.json'
```

随后在新入口加`--response-model 'D:\drone forecast\output\new_response_model.json'`。工具拒绝覆盖旧模型和少于10个合格教师。不要把待评录像混入训练，也不要按录像选择最有利的模型。更多数据并不自动更准：388增加到91教师后的累计389在同102点为41.167px，差于382；因此没有替换默认研究模型。

这是一条有平均误差与尾部收益、尚未稳定达标的研究候选：当前固定图像残差版本13号33.500→34.410、15号45.416→34.298、16号40.531→38.961、11号38.480→30.743、3号68.584→60.701、4号63.361→59.977。全部弱轮廓参考不是独立COM真值；35px内49→57，30px内45→41，不能只宣传均值下降。
