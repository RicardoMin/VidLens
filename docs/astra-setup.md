# 旧 Astra 接入路径

通用配置、项目 skill、CLI 与图片支持说明已移至 [AI 接入](ai-setup.md)。VidLens 使用标准 stdio MCP，可连接支持该协议的 AI 客户端。

旧命令 `python3 scripts/install-astra.py` 保留为兼容入口，接受 `install-ai.py` 的相同参数；默认配置输出目录现为 `integrations/ai`。已有客户端配置无需因脚本改名而重新填写，项目运行路径变化后才需重新生成。
