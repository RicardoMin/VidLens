# VidLens v0.1.0 首发验证

2026-11-03，在 Apple Silicon macOS、Python 3.12 上比较首发前的开发基线与 v0.1.0 实现。下表的文本检索、解码和分页分别计时。

| 路径与条件 | 开发基线 | v0.1.0 |
|---|---:|---:|
| 问题响应的完整 JSON，5 题均值 | 2,185 tokens | 1,740.2 tokens |
| 必需的 10 个证据窗口 | 命中 10 个，最低覆盖 97.7% | 命中 10 个，最低覆盖 100% |
| 已缓存字幕、常驻服务检索 p50，2,500 token 预算 | 7.15 ms | 2.35 ms |
| 已缓存字幕、新 Python 进程 p50，同预算 | 148 ms | 192 ms |
| 600 条字幕的第一页，1,600 token 预算，三次中位数 | 1.032 s | 0.0114 s |
| YouTube 8 秒片段本地首次抽 4 帧 | 0.484 s | 0.191 s |
| B站 8 秒片段本地首次抽 4 帧 | 0.425 s | 0.160 s |

## 文本

固定 3Blue1Brown 神经网络第一讲的 231 条人工中文字幕，全文为 4,842 个估计 token。五道开发题涵盖四个主题，包含两种关于激活函数取值范围的问法。两种预算分别为 2,500 和 4,000，每个问题各测一次新进程和三次常驻服务，产生 40 条记录。评测使用 SQLite 的临时副本，按标注语轨过滤。平时查看画面产生的 OCR 会保留在用户缓存中。

计数器使用 `o200k_base`，覆盖完整工具 JSON。工具声明、图像、模型推理与最终生成属于其他开销。常驻检索受益于索引复用；这次新进程在两种预算下分别测得约 192 ms 和 154 ms，耗时包含解释器和依赖加载。每种预算的新进程 p50 只有五个样本。开发题窗口命中使用 50% 交叠阈值，同时记录每个窗口实际覆盖率和文字锚点。

本轮还检查了混合语轨的一个失败案例。早期优化版本在插入 OCR 后移动了语音边界，仍能命中 ReLU 片段，却漏掉随后给出的函数定义。最终检索按模态、来源和语言独立分组。两种问法都取回了定义原句，固定字幕与混入两条 OCR 的检查都保留了预标注窗口。

另用一条 116 分钟的英文 GPT 编程课抽查。询问 attention 缩放时，首次结果以 1,114 个估计 token 定位到方差解释，但讲者随后才说明 softmax 如何趋向 one-hot。补读 78:00–78:38 再用 1,186 个估计 token 取得该条件。这个样例验证了何时需要展开上下文。[对应讲解](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4654)

## 画面与分页

视觉对照使用两个已经下载的真实片段和一个可重建的幻灯片样本，均关闭 OCR，每次使用新的帧缓存，然后立即读取同一区间。三例的 FFmpeg 调用均从 12 次降至 5 次，缓存 JPEG 从 24 张降至 4 张；热命中约 0.4 ms。预览策略改变后，两条真实片段各有一个分桶的入选时间变化约 0.67 秒。快闪内容仍可能落在采样点之间。另用真实 stdio MCP 读取一段静态视频，四个采样时间点均指向同一张成功解码的图片。

分页样例使用 600 条合成中英字幕和已经缓存的词表。优化前后第一页均返回 21 条、估计 1,475 tokens。长字幕的逐字符恢复、游标失效、部分转写范围、跨进程读取、片段复用及安装配置保留均由离线回归覆盖。

[同机前后对照](../benchmarks/results/iteration.json) · [当前逐次检索结果](../benchmarks/results/reading.json) · [视觉原始计时](../benchmarks/results/visual-sampling.json)

## 复现

完成 README 的安装后，公开课程字幕需要先联网获取一次，后续检索评测只读缓存。若平台更新字幕，可用结果里的语轨 SHA-256 检查是否仍为同一份材料。

```sh
.venv/bin/vidlens --home .vidlens inspect 'https://www.youtube.com/watch?v=aircAruvnKk' --languages zh,en
.venv/bin/python benchmarks/evaluate_reading.py --home .vidlens --output benchmarks/results/reading.json
.venv/bin/python benchmarks/measure_paging.py --home .vidlens
.venv/bin/python benchmarks/measure_visual.py
```

此前的开发基线实测通过了 YouTube 人工字幕、B站局部媒体、macOS Vision OCR 与 Whisper base 中文转写。本轮另验证了 README 的实际命令以及独立进程下的 MCP 图片传输。开发题和单次课程抽查的规模有限，尚无跨产品同题的答案准确率或 Astra 账单对照。[开发基线阅读记录](../benchmarks/results/development-reading-review.json)
