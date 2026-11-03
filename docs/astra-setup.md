# Astra 接入

VidLens 提供项目 skill 和 stdio MCP 两种入口。skill 决定如何阅读，MCP/CLI 提供证据。使用的是当前会话中的 Astra；服务本身不发起额外的大模型请求。

## 先运行本地工具

在项目根目录执行：

```sh
python3 scripts/bootstrap.py --activate
export VIDLENS_HOME="$PWD/.vidlens"
.venv/bin/vidlens doctor --pretty
```

确保 `doctor` 能正常返回。没有 OCR 或 ASR 不影响已有字幕的阅读；`read_frames` 可以在 OCR 不可用时仍返回图像。

## 安装项目 skill

```sh
.venv/bin/python scripts/install-astra.py --project-skill --project-mcp
```

脚本把 skill 放入当前项目的 `.agents/skills/read-video`，并将 MCP 配置加入本项目 `.codex/config.toml`。省略 `--project-mcp` 时仅生成配置片段。已安装 skill 若有不同内容，脚本会停止；检查后用 `--force` 更新。同名 MCP 配置若有差异会保留原设置并提示合并；从不修改全局配置。

在这个项目中新开会话，使宿主发现 `read-video`。可以显式说“用 `$read-video` 阅读这个视频”，也可以直接给 YouTube／B站链接并描述想理解的问题。自动选择取决于宿主是否已重新加载 skill。

当前会话尚未发现 skill 时，可以让 Astra 读取项目里的 `skills/read-video/SKILL.md` 并使用其中的 CLI 路径。skill 自带启动器，从安装位置定位项目环境，不依赖当前工作目录。

## 接入 MCP

安装脚本同时生成两个文件：

- `integrations/astra/codex.toml`：Codex 的 MCP 配置片段。
- `integrations/astra/mcp.json`：支持 `mcpServers` 的宿主配置。

使用 `--project-mcp` 时已完成项目配置写入；重连 MCP 或在受信任的项目中新开会话即可加载。若使用其他宿主，可将生成片段合并到它的 MCP 设置，保留已有服务。

配置使用当前项目 `.venv/bin/python` 的绝对路径，并显式设置 `VIDLENS_HOME`。项目移动后重新运行安装脚本生成路径。服务超时上限为 1500 秒，容纳首次模型准备与本地转写；字幕和缓存检索不会等待这个时限，网络、媒体和转写操作各自还有更短的有界超时。

正常连接后，Astra 应能发现6个默认工具：`read_video`、`read_interval`、`read_frames`、`crop_frame`、`transcribe_video`、`video_coverage`。`read_video` 一次返回可直接分析的原始证据，不需要默认执行多轮重复读取。`read_frames` 将实际图片作为 MCP 内容传入；CLI 路径需要 Astra 另用图片读取工具打开返回文件。

若确实需要额外的导航、独立检索和诊断工具，在已有 `[mcp_servers.vidlens.env]` 段增加 `VIDLENS_ADVANCED_TOOLS = "1"` 并重连。完整模式额外提供 `open_video`、`video_overview`、`find_evidence`、`list_videos`、`doctor`。默认精简模式减少工具声明数量；CLI 命令不受此选项影响。

可用的验证请求：

> 用 VidLens 打开这个链接。先告诉我材料是否完整，再解释作者的实验设置；请核对屏幕上的数字，并附时间定位。

## 检查接入问题

| 现象 | 检查方式 |
|---|---|
| 没有发现工具 | 重连 MCP；检查生成配置的 Python 绝对路径是否存在 |
| 启动失败 | 用同一个 Python 运行 `-m vidlens doctor --home /absolute/cache/path` |
| skill 找不到运行环境 | 重新运行安装脚本；或设置 `VIDLENS_PROJECT` 指向项目目录 |
| 工具有文字但没有图片 | 确认调用的是 `read_frames`；CLI 路径需打开返回图片；检查宿主是否支持 MCP 图片 |
| 链接读取失败 | 根据返回的访问／字幕错误处理；不要把连接正常等同于平台必定可访问 |
| OCR／ASR 不可用 | 查看 `doctor`；OCR 可降级，ASR 需额外安装 `asr` 依赖 |

参考：[Codex MCP 文档](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)、[Skill 文档](https://learn.chatgpt.com/docs/build-skills)。
