# 协作说明（Git 工作流）

这个仓库是「只输入事件相机 RAW、预测无人机未来 0.1/0.2/0.4 秒图像位置」的研究工程。
本文件只讲**怎么和他人一起改代码**；方法、结论、失败记录在根目录 `WORKLOG.md`。

## 仓库边界

| 入库 | 不入库（见 `.gitignore`） |
|---|---|
| 源码（`src/`、`research/stage*/`、`tools/`、`run_*.py`） | 原生 RAW 与视频（`.raw/.avi/.mp4`，单段 0.5–1.9 GB） |
| 文档（`*.md`）与协议/日志中的小 JSON | ONNX 权重（~50 MB，获取方式见 `MODELS.md`） |
| 第三方快照（`external/`，含 LICENSE） | 全部运行输出（`output/`，17+ GB） |
| 测试与复现脚本 | `__pycache__`、`Ultralytics/settings.json`（含本机路径与 uuid） |

**数据集不在仓库里**：自采 RAW 在 `E:\dataset`，FRED 在 `E:\FRED`。克隆后没有数据是正常的，
跑任何实验前先确认数据路径，且不要把它提交进来。

## 首次配置

```powershell
git config user.name  "你的名字"
git config user.email "你的邮箱"
git config core.autocrlf true     # Windows 上避免行尾反复改动
```

## 日常流程

```powershell
git switch -c feat/短描述        # 1. 开分支，不要在 main 上直接改
# ...改代码...
git status                        # 2. 看清楚改了什么
git add 具体文件                   # 3. 按文件加，不要习惯性 git add -A
git commit -m "stage152: 说明这次改了什么、为什么"
git push -u origin feat/短描述    # 4. 推到远端
```

然后开 Pull Request。合并前请别人看一眼 `git diff main`。

## 必须遵守的三条（来自本项目的历史教训）

1. **不修改已冻结的产物**。`research/stage*/` 下的脚本一旦产出了 `protocol.json`/`scores*.json`，
   就不改它本身；要改就新建 `stageNNN` 目录。历史上正是因为改冻结文件，导致结果无法复现。
2. **不美化数字**。负结果、缺失样本、覆盖率都要留在记录里。`output/` 里的报告是证据，
   代码改动必须能被独立重放验证（本项目每个入口都有 `sources_unchanged` 指纹检查）。
3. **权重与数据不进 git**。会用 `.gitignore` 挡住，但提交前仍要 `git status` 确认。

## 提交信息格式

```
stageNNN: 一句话结论

做了什么 / 为什么 / 影响哪个默认路径
验证方式：哪个脚本、哪批数据、多少锚点
负结果：如果没成功，写清楚失败在哪
```

例：

```
stage144: 外推阻尼升为默认

预测位移按 k(h)=1-0.9h^2 收缩，修正 0.4 s 沿航向 +24.5 px 的过外推。
验证：train6 侧视人工框 55.09→50.03、train2 跨录像 66.65→59.30、清晰自洽 30.41→25.60。
负结果：同批数据上"自适应阻尼"与"物理分支融合"均无收益，未采用。
```

## 目录速查

```
event_drone_forecast/
├── src/event_drone_forecast/   主包（几何/感知/运动模型）
├── research/stage1..149/       逐阶段实验，一个阶段一个目录
├── tools/                      报告生成、标注工具、评测脚本
├── run_*.py                    各入口；默认入口是 run_optimized_forecast.py
├── MODELS.md                   权重获取与校验
└── WORKLOG.md                  完整实验记录（本项目最重要的文件）
```

## 冲突怎么办

- 只改自己分支里的文件；`WORKLOG.md` 是最容易冲突的地方——**优先保证不丢别人的记录**，
  冲突时保留双方条目，按日期排序。
- 冻结阶段的目录不要改，冲突基本不会出现在那里。
