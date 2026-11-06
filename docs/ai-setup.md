# AI 接入

VidLens 向 AI 提供字幕、时间定位、章节、画面和可追溯笔记。阅读与讲解由你正在使用的 AI 完成；VidLens 服务本身不另行调用托管大模型。支持标准 **stdio MCP** 的客户端可以接入，核对画面还要求客户端支持 MCP 图片内容。没有 MCP 的环境可以使用 CLI。

## 准备运行环境

在项目根目录运行：

```sh
python3 scripts/bootstrap.py
export VIDLENS_HOME="$PWD/.vidlens"
.venv/bin/python -m vidlens doctor --pretty
```

默认准备本地 ASR 依赖；仅需已有字幕可用 `--captions-only`。首次转写可能需要下载模型。OCR／ASR 能否使用以 `doctor` 的结果为准；没有 OCR 时仍可返回图片供 AI 核对。

Apple Silicon 的可选安装：

```sh
python3 scripts/bootstrap.py --apple-asr
```

这会安装 `asr-mlx` extra，包含 MLX Whisper 与用于语音检测的 faster-whisper；不会下载模型，权重在首次转写时取得并校验。`doctor.transcription` 和章节计划的 `transcription` 返回可用后端及建议配置。显式 `backend="mlx-whisper"` 默认使用 `large-v3-turbo`，`backend="faster-whisper"` 默认仍为 CPU `base`；工具默认值保持兼容，skill 根据阅读计划选择已安装的后端。

这一步生成本地接入文件，**不修改客户端全局设置**。本仓库脚本使用 `.venv/bin/python`，适用于 macOS／Linux 或 WSL 中的项目环境。

## 连接 MCP 客户端

```sh
.venv/bin/python scripts/install-ai.py
```

默认生成 `integrations/ai/mcp.json` 和 `integrations/ai/codex.toml`，包含项目 Python 和缓存的绝对路径。可以用 `--output` 改变输出目录，或用 `--home` 选择缓存位置。

支持 `mcpServers` 配置格式的客户端可以合并以下条目；字段名称和配置位置仍以客户端自己的说明为准：

```json
{
  "mcpServers": {
    "vidlens": {
      "command": "/absolute/path/VidLens/.venv/bin/python",
      "args": ["-m", "vidlens", "mcp"],
      "env": {
        "VIDLENS_HOME": "/absolute/path/VidLens/.vidlens",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

`mcpServers` 是常见客户端配置约定，不是所有 MCP 客户端统一的配置文件格式。服务使用标准 stdio 协议；客户端还应能向 AI 传递工具返回的图片，才能完成图表、代码和屏幕数字的视觉核对。

Codex 的对应配置片段为：

```toml
[mcp_servers.vidlens]
command = "/absolute/path/VidLens/.venv/bin/python"
args = ["-m", "vidlens", "mcp"]
tool_timeout_sec = 1500

[mcp_servers.vidlens.env]
VIDLENS_HOME = "/absolute/path/VidLens/.vidlens"
PYTHONUNBUFFERED = "1"
```

安装脚本可以显式写入**当前项目**的 Codex 配置及 skill：

```sh
.venv/bin/python scripts/install-ai.py --project-skill --project-mcp
```

`--project-mcp` 合并到 `.codex/config.toml`，保留其他服务和设置；`--project-skill` 将 `skills/read-video` 复制到 `.agents/skills/read-video`。这些选项面向支持相应项目目录约定的客户端。也可以运行 `python3 scripts/bootstrap.py --activate` 一次完成项目激活。同名 MCP 配置有差异时停止并保留原内容；skill 有差异时先检查，再用 `--force` 明确更新。任何模式都不写入全局配置。

重连 MCP 或在项目中新开会话，让客户端重新发现工具与 skill。其他客户端可读取 `skills/read-video/SKILL.md`，或按自己的 skill 安装方式使用该目录。项目移动后重新生成配置。旧 `scripts/install-astra.py` 保留为兼容入口，接受相同参数并调用新安装器，默认输出也改为 `integrations/ai`。

## 从读取到讲解

总结全片时，AI 应调用 `read_video(url, intent="summary")` 获得 `study_plan`。这一步只提供章节计划；AI 还需依据 `sections` 逐章调用 `read_section`，读完每章的分页文本并核对实际图片。章节地图有分页时用 `study_status` 继续。`scan_video` 提供全片候选画面地图；章节的 `frame_cursor` 可继续读取候选序列，帮助补看代码、图表等变化。具体问题使用 `intent="question"`，围绕相关证据展开。

`save_note` 保存带原始证据 ID 的章节笔记；`search_knowledge` 检索已保存的概念与关系。需要延伸时，由宿主 AI 搜索并阅读外部主源，再用 `add_reference` 保存来源及与章节的关联。缓存里的旧笔记不能代替未读的新证据。

`scan_video` 默认返回 `navigation` 视图，`view="full"` 保留全部原始候选。两种视图各有自己的游标，续页要带上返回的 `view`。章节图片续页使用 `frame_view`，与文本游标独立；首批代表图仍按稳定时间覆盖选取。精简导航用于减少重复查看，不判断一张图的概念意义。

围绕具体问题选图时，先读相关原文，再调用 `read_frames(..., question="用户的问题", visual_queries=["更新前的状态", "中间计算", "最终结果"])`。工具结合原文与屏幕 OCR 查找相关状态，并补查相邻变化，最多探查 96 个候选后返回指定数量的图片。`visual_queries` 可省略，也可由 AI 按问题拆成至多六个取证目标。OCR 不可用时明确回退到时间覆盖；实际图片仍需检查。该模式也可通过 `read_section(question=...)` 使用，与 `frame_cursor` 分开调用。

手写公式、树形结构与箭头关系适合增加一次视觉选择：`read_frames(..., question=..., preview=True)` 返回带编号的候选预览；AI 看过预览后用 `read_frames(..., timestamps=[...])` 取回选中的原图。预览保留问题相关状态和时间覆盖，帮助发现纯 OCR 排序遗漏的中间步骤。这一步额外占用图片上下文；缩略预览与原图交付分别处理。

图表数值决定答案时，调用 `check_facts(video_id, claims)` 检查字段、数值及单位在同一画面行中的对应关系，再实际看返回的 PNG 裁剪。每项包含 `id/timestamp/label/value/unit/scope/meaning`，可加归一化 `box`；值使用数字字符串，支持科学计数和千分位。`scope/meaning` 是 AI 的待核对陈述，工具不据此认定语义正确。完整步骤和 JSON 示例见[图表核对](../skills/read-video/references/fact-checking.md)。更新服务后需重连 MCP 才能发现新工具。

图中小字不清时，调用 `read_detail(video_id, timestamp, boxes=[[x,y,w,h]])`，同时查看上下文与源像素局部块。可加 `upgrade_source=True` 尝试更高画质；实际取得的尺寸在 `detail.source_size`，请求档位在 `acquisition`。平台只提供低清源时仍会保留这个事实。CLI 对应 `detail VIDEO_ID TIMESTAMP --box X Y W H --upgrade-source`。

可用的验证请求：

> 用 VidLens 逐章读这个视频，讲清作者从哪个问题出发、方法如何工作、实验验证了什么。核对相关代码和图表，并附时间定位。

技术过程适合逐步演示时，skill 附带的 `build_lesson.py` 可将整理后的 JSON 生成可离线打开的交互 HTML。它支持字幕、解释、状态卡片和键盘切换，详见 [交互讲解资源](../skills/read-video/references/interactive.md)。

## CLI 也可使用

```sh
.venv/bin/python -m vidlens inspect 'VIDEO_URL' --intent summary
.venv/bin/python -m vidlens section VIDEO_ID SECTION_ID --max-tokens 6000
.venv/bin/python -m vidlens scan VIDEO_ID
.venv/bin/python -m vidlens export VIDEO_ID --output lesson.md
```

CLI 返回的图片路径需要使用图片工具实际打开。已安装 skill 的 `scripts/vidlens.py` 启动器可从任意工作目录定位项目环境；详见 [阅读与笔记流程](../skills/read-video/references/workflow.md)。

## 排查连接

| 现象 | 检查方式 |
|---|---|
| 工具未出现 | 重连 MCP，确认配置中的 Python 绝对路径存在 |
| 启动失败 | 用同一个 Python 执行 `-m vidlens --home /absolute/cache/path doctor --pretty` |
| skill 找不到环境 | 重新安装 skill，或设置 `VIDLENS_PROJECT` 指向项目目录 |
| 有文字没有图片 | 确认客户端支持 MCP 图片；CLI 必须实际打开返回的图片路径 |
| 链接读取失败 | 按返回的网络、访问或字幕错误处理；MCP 连通不代表平台必定可访问 |
| 转写不可用 | 检查 ASR 依赖与模型准备情况；初次模型下载不同于缓存读取 |

Codex 配置的工具超时为 1500 秒，用于容纳首次准备和本地转写；网络和媒体操作仍有各自的有界超时。其他客户端可能需要在自身设置中提高工具超时。
