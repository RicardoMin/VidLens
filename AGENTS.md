# VidLens workspace

默认使用中文交流。这个工作区为 AI 客户端提供 YouTube／B站视频阅读与学习工具。

用户给出视频链接并要求理解、总结、学习或问答时，先读取
`skills/read-video/SKILL.md`，使用已接入的 VidLens MCP `read_video`。
若当前会话尚未加载 MCP，总结使用项目 `.venv/bin/python -m vidlens --home .vidlens inspect URL --intent summary`；
具体问题使用 `--intent question --question QUESTION`。总结按返回章节继续读取原文和图片。
CLI 返回的图片需要实际通过图片工具打开后才能声称核对过画面。

开发时使用 `.venv/bin/python` 与 `.venv/bin/ruff`。测试不默认联网；真实平台测试须区分
首次获取、缓存、字幕、OCR 和 ASR。`benchmarks/results` 保存可复现指标；`.vidlens` 是私有运行缓存。
不得把本地 tokenizer 估计称为模型实际账单。阅读进度分别记录文字交付、画面采样和学习笔记。

保留用户未完成的工作；未经用户明确要求不提交、推送或发布。
