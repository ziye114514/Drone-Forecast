# 怎么让别人进来一起做

这份不是给人随便看看的流程文档，是**你现在就能照着做**的清单。分三块：
仓库要补什么、人去哪儿找、找到人之后第一件事让他们做什么。

---

## 一、仓库这边还差什么

现状：仓库是 public、MIT、issues 开着、2044 文件、新克隆 25 个单测能过。
已经能让人进来。但有几处会让人找不到下手的地方。

### 1. 填仓库描述和 topics（2 分钟，收益最大）

现在 `description` 和 `topics` 都是空的。别人在 GitHub 搜 `event camera` +
`trajectory prediction` 找不到你。

去 https://github.com/ziye114514/Drone-Forecast/settings 或者在仓库首页点
⚙️ 旁边的 Edit：

**Description**（一句话，别写成摘要）：

```
RAW-only event-camera drone tracking and 0.1/0.2/0.4 s image-plane forecasting (C++/CUDA research code)
```

**Topics**（这些是搜索入口，填满 20 个）：

```
event-camera  neuromorphic  drone  uav  trajectory-prediction  motion-forecasting
computer-vision  onnx  cuda  python  real-time  object-tracking  kalman-filter
metavision  prophesee  robotics  perception  event-based-vision
```

### 2. 已经帮你补好的（这次提交里）

| 文件 | 作用 |
|---|---|
| `.github/ISSUE_TEMPLATE/result_report.md` | 报告一次可复现测量，强制填「口径」「负结果」 |
| `.github/ISSUE_TEMPLATE/environment.md` | 跑不起来时提供完整信息，省掉来回问 |
| `.github/ISSUE_TEMPLATE/idea.md` | 提方向，强制先看已证伪清单 |
| `.github/pull_request_template.md` | PR 前自查：冻结产物、口径、权重 |
| `.github/workflows/smoke.yml` | 每次 push/PR 自动跑 25 个单测 + 语法 + 平铺名解析 |
| `tools/check_flat_names.py` | 上面那条 CI 用的检查器，本地也能跑 |

CI 里那条**平铺名解析**检查是专门为这个仓库写的：`core/` 里 340 个模块靠裸名
import 互相找，一旦谁搬了文件，`tools/prepare_github_release.py` 的 COPY 表
和几个快照工具的 SOURCES 表会静默失效——这次搬家就是这么坏的。

### 3. 三个 release（可选，但很有用）

别人最想先看到的是**能跑的东西**，不是 340 个模块。你有现成的发布打包脚本：

```powershell
D:\drone forecast\.venv-event311\Scripts\python.exe tools\prepare_github_release.py `
  --output 'D:\drone forecast\output\release_v040a2'
```

刚验过：成功产出 170 文件 / 6.87 MB，包内 `forecast.py --help`、
`tools/demo_synthetic.py` 都能跑通，`.github/workflows/synthetic.yml` 也在包里。

打到 GitHub Releases 上，标签用 `v0.4.0-alpha.2`，说明里写清「能跑的合成控制 +
不能跑真实 RAW（数据不在包里）」。

---

## 二、人去哪儿找

按**成功率**排序，不是按听起来高大上排：

### 1. 你自己的实验室 / 师兄师姐（成功率最高）

最实际的一条。这个项目本来就是保研材料，同组的人知道你的约束（没有飞控真值、
只有一路事件相机），不用花两小时解释背景。

具体怎么说：

> 我在做一个只用事件相机预测无人机未来位置的工程，代码整理好了，有几个方向
> 我一个人试不过来——侧视段的桨叶关联和跨录像泛化。有没有兴趣挑一个试试？
> 数据和权重我这边有，能给你一份。

**关键：给对方一个具体的、边界清楚的小任务，不是"来一起做项目"。**

### 2. Prophesee / Metavision 社区

你用的是 Metavision SDK，那边有官方论坛和 GitHub 组织。他们关心的是
**SDK 用得对不对、有没有新玩法**，不关心你的预测精度。

合适的切入点：你手上有自采的 RAW、有侧视退化的观测数据、有纯桨叶前端的实现。
这些对他们是**真实数据反馈**。

- Metavision 官方论坛：https://support.prophesee.ai/
- 提一个 issue 或帖子，标题类似「Rotor-only front-end on side-view: when does
  the shared-arc assumption break?」，附上你的观测（不要附 RAW，附统计）。

### 3. 事件相机 / 神经形态视觉的研究者

搜这些关键词找正在做的人：`event camera UAV`、`event-based motion prediction`、
`neuromorphic drone`。看 Recent 的论文，作者通常有 GitHub。

**别群发。** 挑 2–3 个真正做这块的，一封一封写。说清楚：
- 你只有一路相机、没有 IMU 真值（这是他们的常见场景，不是缺陷）
- 你已经证伪了什么（他们最想知道的）
- 你需要什么（具体到「帮我看看侧视段的频率估计为什么质量只有 0.375」）

### 4. 开源社区（成功率最低但覆盖面最大）

在 README 顶部加一行醒目的：

```markdown
> **找合作者**：侧视段桨叶关联、跨录像泛化两个方向缺人手。数据与权重可提供。
> 见 [Issues](https://github.com/ziye114514/Drone-Forecast/issues) 里标 `help wanted` 的条目。
```

然后**真的去开那几个 issue**，标上 `help wanted` + `good first issue`。
空仓库喊"求合作"没人理，有具体 issue 的才有人点。

---

## 三、人来了之后，第一件事

这一步决定他会不会留下。

### 别让他 clone 完就卡住

新人第一个小时大概率卡在：没数据、没权重、`--help` 报错。

所以第一封回复里直接给：

1. **能跑的最小例子**（不需要数据）：
   ```powershell
   python tools\demo_synthetic.py --output D:\tmp\demo
   ```
   这个刚验过能跑，输出 `Synthetic controls passed`。

2. **25 个单测怎么跑**：
   ```powershell
   python -m unittest discover -s tests
   ```
   注意提醒：**环境里没有 pytest**，只能 unittest。

3. **数据怎么拿**：说明自采 RAW 在你本机 `E:\dataset`（train1–6），FRED 在 `E:\FRED`。
   要么给他一份，要么告诉他哪几段最关键（train6 侧视 110–119.5s 是主基准）。

4. **权重怎么拿**：`MODELS.md` 里有 5 个 ONNX 的 SHA256。给他 `v1` 那个就够
   （`live_position_accumulated_v1.onnx`）。

### 给他一个能在一两天内做完的事

不要给"改进预测精度"这种。给这种：

- ✅ 「用 `research/stage144/score_dense.py` 复现一下 50.03 px，看你能不能得到一样的结果」
  —— 这是复现，有明确终点，做完了他就理解了整条链。
- ✅ 「`tools/check_flat_names.py` 现在覆盖 7 个列表，你看看 `tools/` 下还有没有别的
  文件也是这种裸名列表」（刚做完，是个真实的小活）
- ✅ 「train2 那段 36 个标注帧，帮我把 0.4s 的误差按横向/沿航向拆一下」
- ❌ 「试试用 Transformer」—— 没边界，做不完，做完了也不知道算不算成功。

### 告诉他规矩

`CONTRIBUTING.md` 里三条，但要口头强调一遍：

> 已经跑出结果的脚本不要改，要改就新建 `stageNNN` 目录。
> 数字不能美化，负结果照写。
> 权重和 RAW 别提交。

这三条不是为了管人，是因为**这个项目的历史价值在失败记录里**——四轮物理分支、
速度守门、自适应阻尼全部实测无增益，这些留在 `research/` 里就是为了让新人别重走。

---

## 四、话术模板

### 发给认识的人

> 我在做一个事件相机预测无人机位置的工程（保研材料）。整理成开源了：
> https://github.com/ziye114514/Drone-Forecast
>
> 现在的状态是：平稳飞行段 0.4 秒误差 50 px，侧视段 59 px，30 FPS 没做到。
> 有一些方向我一个人试不过来。
>
> 你有没有兴趣挑一个小块？比如侧视段的桨叶关联——现有方案在同 tick 上有 34%
> 的时刻会分出两个频率族，我一直没搞清那是真的还是估计噪声。数据和权重我能给你。

### 发到公开渠道

> **RAW-only event-camera drone forecasting** — 只用一路事件相机的 ON/OFF 事件流，
> 不做灰度重建，不预知目标参数，预测未来 0.1/0.2/0.4 秒的图像位置。
>
> 已整理为开源研究代码：https://github.com/ziye114514/Drone-Forecast
>
> 诚实的现状：平稳段 0.4s ≈ 50 px，侧视段 ≈ 59 px，没到 30 px 目标；
> 实测 73.8 ms/帧，没到 30 FPS。已证伪的方向都留在 `research/` 里
> （速度守门、四路物理修正、平移/摆动分离、自适应阻尼、CTRV/IMM）。
>
> 缺人的地方：侧视段桨叶关联、跨录像泛化。数据和权重可以提供给认真做的人。

**注意最后一句**。"可以提供给认真做的人"比"欢迎交流"有效十倍——前者筛掉了
随手 star 的，后者招来一堆问"怎么跑"的。

---

## 五、别做的事

- **别在群里刷链接**。发一次，没人应就算了，隔两周换个角度再说一次。
- **别承诺精度**。现在是 50 px 不是 30 px，说 30 会被打脸。
- **别把 RAW 传上去**。17.7 GB，也不该公开（里面有你的拍摄环境）。
- **别为了显得厉害删掉失败记录**。这个仓库最值钱的部分恰恰是那些失败。
