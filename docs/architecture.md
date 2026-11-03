# 一次读取如何完成

Astra 从 MCP 的 `read_video(url, question)` 获取原始证据。CLI 的 `inspect` 调用同一个 Python 入口。两者都由当前对话里的 Astra 解释结果，阅读策略在 [SKILL.md](../skills/read-video/SKILL.md)。

| 边界 | 职责 |
|---|---|
| [VideoService](../src/vidlens/service.py) | 协调平台获取、转写与视觉操作。已解析过的短链接可离线命中。并发写入同一视频时按内容加锁。 |
| [SourceProvider](../src/vidlens/sources.py) | 识别平台与 B 站分 P，取得字幕，按需下载音轨或视频区间，必要时调用本地 Whisper。 |
| [Store](../src/vidlens/store.py) | SQLite 保存视频、带时间的原文、OCR 和已交付范围。一次读事务取得同一版本的材料。 |
| [EvidenceReader](../src/vidlens/reading.py) | 保留当前视频的阅读快照，按需建立 [BM25 倒排索引](../src/vidlens/retrieval.py)。CLI 或 MCP 修改证据后，下一次读取重建快照。 |
| [TokenBudget](../src/vidlens/budget.py) | 对完整 JSON 计量。区间阅读保留可继续读取的前缀，低预算通读从整条时间轴均匀选段，超长单条字幕可从字符位置继续。 |
| [MediaCache](../src/vidlens/media_cache.py) | 按视频版本、音视频种类和分辨率设置查找能覆盖请求的最短已下载片段。成功下载的媒体才会进入可复用目录。 |
| [VisualReader](../src/vidlens/visual.py) | 局部预览、候选选择、高清帧、OCR 与裁剪。采样时间始终相对于原视频。MCP 使用 `image_index` 将采样时间对应到实际传输的图片。 |

## 有具体问题时

1. 取可用字幕并保留其来源与时间。先在本地计算检索分数。
2. 在命中的段落中选择相邻原句，保留回答所需的上下文。简体查询也可命中繁体字幕，返回的引文保留原始字形。
3. 第一次调用直接返回这些原句。如果因果解释仍在下一句，按时间展开附近的原文；如果答案依赖代码或图表，再取局部画面。

## 通读与补证

不带问题时，优先返回预算内的全文。超出预算时，`sampled_transcript` 明确标记采样结果。只有 OCR 或元数据时，分别返回 `visual_evidence` 或 `no_transcript`。局部转写的范围由 `speech_scope` 给出。分页游标绑定视频、区间与证据指纹，材料变化后应从头请求该区间。

画面和音轨独立获取。例如，已经下载第 60–90 秒的画面，后续读取第 67–70 秒可以从同一文件解码。需要口述内容时单独获取音轨。OCR 是画面在指定瞬间的机器读数，核对公式符号、小字或基线时，可裁剪对应时间的原帧。
