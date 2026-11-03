"""Bounded, caption-first acquisition of YouTube and Bilibili evidence."""

from __future__ import annotations

import http.cookiejar
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import ParseResult, parse_qs, urljoin, urlparse
from urllib.request import getproxies

import httpx

from .models import AcquiredVideo, MediaAsset, Segment, Video, VidLensError
from .subtitles import parse_subtitles

_YT_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
_BILI_HOSTS = {"bilibili.com", "www.bilibili.com", "m.bilibili.com"}
_SHORT_HOSTS = {"b23.tv", "www.b23.tv"}
_CAPTION_DOMAINS = (
    "youtube.com",
    "googlevideo.com",
    "bilibili.com",
    "hdslb.com",
    "bili.com",
    "bilivideo.com",
)
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
)
_MAX_RESPONSE = 12 * 1024 * 1024


def _validated_url(url: str) -> ParseResult:
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError
        if parsed.port not in {None, 80, 443} or not parsed.hostname:
            raise ValueError
        return parsed
    except (ValueError, AttributeError) as exc:
        raise VidLensError(
            "INVALID_URL", "Provide a public YouTube or Bilibili video URL."
        ) from exc


def normalize_url(url: str) -> str:
    """Canonical URL; Bilibili's part number is part of its content identity."""
    parsed = _validated_url(url)
    if parsed.hostname in _SHORT_HOSTS:
        try:
            with httpx.Client(
                timeout=12, follow_redirects=False, headers={"User-Agent": _USER_AGENT}
            ) as client:
                for _ in range(5):
                    if parsed.hostname not in _SHORT_HOSTS:
                        break
                    response = client.head(parsed.geturl())
                    if not response.is_redirect or not response.headers.get("location"):
                        raise VidLensError(
                            "SHORT_LINK",
                            "The b23.tv link did not resolve to a video.",
                            "Use the full bilibili.com/video/... URL.",
                        )
                    parsed = _validated_url(
                        urljoin(str(response.url), response.headers["location"])
                    )
        except httpx.HTTPError as exc:
            raise VidLensError(
                "NETWORK",
                "Could not resolve the b23.tv short link.",
                "Check connectivity or paste the full Bilibili video URL.",
            ) from exc
    query = parse_qs(parsed.query)
    host, path = parsed.hostname, parsed.path.strip("/")
    if host in _YT_HOSTS:
        if host.endswith("youtu.be"):
            identity = path.split("/")[0]
        elif path == "watch":
            identity = query.get("v", [""])[0]
        elif path.startswith(("shorts/", "embed/", "live/")):
            identity = path.split("/")[1]
        else:
            identity = ""
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", identity):
            return f"https://www.youtube.com/watch?v={identity}"
    elif host in _BILI_HOSTS:
        match = re.fullmatch(r"video/(BV[A-Za-z0-9]{10}|av\d+)", path, re.IGNORECASE)
        if match:
            identity = match[1]
            identity = "BV" + identity[2:] if identity[:2].lower() == "bv" else identity.lower()
            try:
                part = int(query.get("p", ["1"])[0])
            except ValueError as exc:
                raise VidLensError(
                    "INVALID_PART", "Bilibili part must be a positive integer."
                ) from exc
            if not 1 <= part <= 10000:
                raise VidLensError("INVALID_PART", "Bilibili part must be between 1 and 10000.")
            return f"https://www.bilibili.com/video/{identity}/?p={part}"
    raise VidLensError(
        "UNSUPPORTED_URL",
        "This URL is not a supported individual YouTube or Bilibili video.",
        "Paste a watch, youtu.be, shorts, or bilibili.com/video URL; playlists and channel pages are not video inputs.",
    )


def video_identity(url: str) -> str:
    canonical = normalize_url(url)
    parsed = urlparse(canonical)
    query = parse_qs(parsed.query)
    if parsed.hostname == "www.youtube.com":
        return f"youtube:{query['v'][0]}"
    return f"bilibili:{parsed.path.strip('/').split('/')[-1]}:p{query['p'][0]}"


def _language_rank(language: str, requested: tuple[str, ...]) -> int:
    language = language.lower().replace("_", "-").removeprefix("ai-")
    for index, target in enumerate(requested):
        target = target.lower().replace("_", "-")
        if language == target or language.startswith(target + "-"):
            return index
    return len(requested) + 1


class SourceProvider:
    def __init__(self, cache_dir: Path, cookie_file: Path | None = None):
        self.cache_dir = Path(cache_dir)
        self.cookie_file = Path(cookie_file).expanduser() if cookie_file else None
        if self.cookie_file and not self.cookie_file.is_file():
            raise VidLensError(
                "COOKIE_FILE",
                "The explicitly configured cookie file does not exist.",
                "Provide an existing Netscape cookies file, or omit cookie_file.",
            )

    def _client(self) -> httpx.Client:
        cookies = None
        if self.cookie_file:
            jar = http.cookiejar.MozillaCookieJar(str(self.cookie_file))
            try:
                jar.load(ignore_discard=True, ignore_expires=False)
            except (OSError, http.cookiejar.LoadError) as exc:
                raise VidLensError(
                    "COOKIE_FILE",
                    "Could not read the configured cookie file.",
                    "Use Netscape/Mozilla cookie-file format.",
                ) from exc
            cookies = jar
        return httpx.Client(
            timeout=httpx.Timeout(15, connect=8),
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT, "Referer": "https://www.bilibili.com/"},
            cookies=cookies,
        )

    def _get(self, url: str, params: dict | None = None) -> str:
        started = time.monotonic()
        parsed = _validated_url(url)
        if not any(
            parsed.hostname == domain or parsed.hostname.endswith("." + domain)
            for domain in _CAPTION_DOMAINS
        ):
            raise VidLensError(
                "SOURCE_HOST",
                "The platform returned an unsupported caption host.",
                "Update VidLens or import an exported subtitle file.",
            )
        try:
            with self._client() as client:
                for _ in range(4):
                    with client.stream("GET", url, params=params) as response:
                        if response.is_redirect:
                            url = urljoin(str(response.url), response.headers.get("location", ""))
                            target = _validated_url(url)
                            if not any(
                                target.hostname == d or target.hostname.endswith("." + d)
                                for d in _CAPTION_DOMAINS
                            ):
                                raise VidLensError(
                                    "SOURCE_HOST",
                                    "A caption redirect left the supported platform hosts.",
                                )
                            params = None
                            continue
                        response.raise_for_status()
                        chunks: list[bytes] = []
                        length = 0
                        for chunk in response.iter_bytes():
                            if time.monotonic() - started > 45:
                                raise VidLensError(
                                    "SOURCE_TIMEOUT",
                                    "The platform response took too long.",
                                    "Retry on a faster connection or import local subtitles.",
                                )
                            length += len(chunk)
                            if length > _MAX_RESPONSE:
                                raise VidLensError(
                                    "RESPONSE_LIMIT",
                                    "The platform response exceeded the 12 MiB caption limit.",
                                    "Read a shorter video or import a local subtitle file.",
                                )
                            chunks.append(chunk)
                        return b"".join(chunks).decode("utf-8-sig", errors="replace")
        except httpx.HTTPError as exc:
            raise VidLensError(
                "SOURCE_ACCESS",
                "The video platform could not be reached or refused this request.",
                "Check network access. Restricted videos may require an explicitly supplied cookie file; a local subtitle import also works.",
            ) from exc
        raise VidLensError("SOURCE_REDIRECT", "The platform returned too many redirects.")

    def _json(self, url: str, params: dict | None = None) -> dict:
        try:
            result = json.loads(self._get(url, params))
            if not isinstance(result, dict):
                raise TypeError
            return result
        except (ValueError, TypeError) as exc:
            raise VidLensError(
                "SOURCE_RESPONSE",
                "The platform returned an unexpected response.",
                "Retry later or update yt-dlp if the platform API changed.",
            ) from exc

    def _run(self, command: list[str], timeout: int = 60, input: str | None = None) -> str:
        environment = os.environ.copy()
        # urllib/yt-dlp honor macOS system proxies, while FFmpeg only reads env.
        # Forward the already configured route without logging proxy credentials.
        for scheme in ("http", "https"):
            proxy = getproxies().get(scheme)
            if proxy and urlparse(proxy).scheme in {"http", "https"}:
                environment.setdefault(f"{scheme}_proxy", proxy)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                start_new_session=os.name == "posix",
            )
            try:
                stdout, stderr = process.communicate(input=input, timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.communicate()
                raise
        except subprocess.TimeoutExpired as exc:
            raise VidLensError(
                "SOURCE_TIMEOUT",
                f"Video source processing exceeded {timeout} seconds.",
                "Retry on a faster connection, use a smaller interval, or import local subtitles/media.",
            ) from exc
        except OSError as exc:
            raise VidLensError(
                "SOURCE_EXEC",
                "A required video processing executable could not start.",
                "Run vidlens doctor and reinstall dependencies.",
            ) from exc
        if process.returncode:
            diagnostic = stderr.lower()
            if "vidlens_no_audio" in diagnostic:
                raise VidLensError(
                    "ASR_NO_AUDIO",
                    "The selected media asset has no audio stream.",
                    "Download the interval with audio_only=True or import a media file containing audio.",
                )
            if len(command) > 3 and command[2:4] == ["vidlens.sources", "whisper"]:
                if "no module named" in diagnostic:
                    raise VidLensError(
                        "ASR_UNAVAILABLE",
                        "The optional local speech-recognition dependencies are not installed.",
                        "Install VidLens with its asr extra (uv sync --extra asr or pip install 'vidlens-astra[asr]').",
                    )
                if any(
                    term in diagnostic
                    for term in (
                        "huggingface_hub",
                        "snapshot_download",
                        "localentrynotfounderror",
                        "vidlens_asr_model_download",
                    )
                ):
                    raise VidLensError(
                        "ASR_MODEL_DOWNLOAD",
                        "The local Whisper model could not be downloaded from Hugging Face.",
                        "Check model-source connectivity and your configured proxy, then retry. You can also pass an existing local CTranslate2 Whisper model directory.",
                    )
                raise VidLensError(
                    "ASR_FAILED",
                    "Local speech recognition failed while loading the model or decoding audio.",
                    "Check that the audio is readable and the selected Whisper model is supported; use a smaller model if memory is limited.",
                )
            if "no module named" in diagnostic:
                hint = "Install VidLens dependencies; speech transcription additionally requires the 'asr' extra."
            elif any(
                term in diagnostic
                for term in ("sign in", "login", "cookies", "bot", "403", "412", "blocked")
            ):
                hint = "The platform blocked anonymous access. Supply a cookie file explicitly if you have access, or import local subtitles/media."
            elif "requested format" in diagnostic:
                hint = "The requested media format is unavailable. Update yt-dlp or import a local media file."
            else:
                hint = "Check connectivity and update yt-dlp. Local subtitle/media import can be used when platform access fails."
            # Platform diagnostics may contain signed URLs or cookie paths.
            raise VidLensError("SOURCE_FAILED", "The video source operation failed.", hint)
        if len(stdout) > 32 * 1024 * 1024:
            raise VidLensError("RESPONSE_LIMIT", "The video source returned too much data.")
        return stdout

    def _yt_base(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--no-playlist",
            "--no-warnings",
            "--socket-timeout",
            "12",
            "--retries",
            "1",
            "--extractor-retries",
            "1",
            "--no-progress",
        ]
        if self.cookie_file:
            command += ["--cookies", str(self.cookie_file)]
        return command

    def _yt_info(self, url: str) -> dict:
        output = self._run(
            self._yt_base() + ["--dump-single-json", "--skip-download", "--", url], timeout=60
        )
        try:
            result = json.loads(output)
            if not isinstance(result, dict) or result.get("_type") == "playlist":
                raise ValueError
            return result
        except (ValueError, TypeError) as exc:
            raise VidLensError(
                "SOURCE_RESPONSE", "yt-dlp did not return a single video metadata object."
            ) from exc

    def _captions_from_info(self, info: dict, languages: tuple[str, ...]) -> list[Segment]:
        candidates = []
        for source, field in (("manual", "subtitles"), ("automatic", "automatic_captions")):
            for language, tracks in (info.get(field) or {}).items():
                if language == "live_chat":
                    continue
                for track in tracks:
                    ext = track.get("ext", "")
                    if ext in {"json3", "vtt", "srt", "json"} and (
                        track.get("url") or track.get("data")
                    ):
                        candidates.append(
                            (
                                _language_rank(language, languages),
                                source != "manual",
                                {"json3": 0, "json": 1, "vtt": 2, "srt": 3}[ext],
                                language,
                                source,
                                track,
                            )
                        )
        for _, _, _, language, source, track in sorted(candidates, key=lambda c: c[:3])[:6]:
            try:
                content = track.get("data") or self._get(track["url"])
                result = parse_subtitles(content, track["ext"], source, language)
                if result:
                    return result
            except VidLensError:
                continue
        return []

    def _youtube_transcript(self, identity: str, languages: tuple[str, ...]) -> list[Segment]:
        output = self._run(
            [sys.executable, "-m", "vidlens.sources", "youtube", identity, json.dumps(languages)],
            timeout=40,
        )
        try:
            data = json.loads(output)
            return parse_subtitles(
                json.dumps(data["segments"]), "json", data["source"], data["language"]
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise VidLensError(
                "SUBTITLE_PARSE", "The YouTube transcript response could not be decoded."
            ) from exc

    def acquire(self, url: str, languages: tuple[str, ...] = ("zh", "en")) -> AcquiredVideo:
        started = time.monotonic()
        canonical = normalize_url(url)
        if "bilibili.com" in urlparse(canonical).hostname:
            result = self._acquire_bilibili(canonical, languages)
        else:
            result = self._acquire_youtube(canonical, languages)
        result.timings["source_seconds"] = round(time.monotonic() - started, 3)
        if not result.segments:
            result.warnings.append(
                "No usable captions were retrieved. This is metadata only; use explicit transcription or read relevant frames before making content claims."
            )
        return result

    def _acquire_youtube(self, url: str, languages: tuple[str, ...]) -> AcquiredVideo:
        identity = parse_qs(urlparse(url).query)["v"][0]
        warnings: list[str] = []
        info: dict = {}
        try:
            info = self._yt_info(url)
        except VidLensError as exc:
            warnings.append(f"Full metadata unavailable ({exc.code}).")
        if not info:
            try:
                embed = self._json("https://www.youtube.com/oembed", {"url": url, "format": "json"})
                info = {
                    "title": embed.get("title", identity),
                    "uploader": embed.get("author_name", ""),
                }
            except VidLensError:
                pass
        segments = self._captions_from_info(info, languages)
        if not segments:
            try:
                segments = self._youtube_transcript(identity, languages)
            except VidLensError as exc:
                warnings.append(f"Caption fallback unavailable ({exc.code}). {exc.hint}")
        if not info and not segments:
            raise VidLensError(
                "VIDEO_UNAVAILABLE",
                "Could not retrieve this YouTube video or its captions.",
                "Confirm that the video is accessible. Check the network, supply cookies explicitly for authorized restricted content, or import local evidence.",
            )
        duration = float(info.get("duration") or 0)
        metadata = {"author": info.get("uploader", ""), "duration_known": duration > 0}
        video = Video(
            f"youtube:{identity}",
            url,
            "youtube",
            info.get("title") or identity,
            duration,
            self._chapters(info),
            segments[0].language if segments else str(info.get("language") or ""),
            str(info.get("description") or "")[:12000],
            metadata,
        )
        return AcquiredVideo(video, segments, warnings)

    def _acquire_bilibili(self, url: str, languages: tuple[str, ...]) -> AcquiredVideo:
        identity = urlparse(url).path.strip("/").split("/")[-1]
        part = int(parse_qs(urlparse(url).query)["p"][0])
        warnings: list[str] = []
        params = {"bvid": identity} if identity.startswith("BV") else {"aid": identity[2:]}
        try:
            payload = self._json("https://api.bilibili.com/x/web-interface/view", params)
            if payload.get("code") != 0 or not isinstance(payload.get("data"), dict):
                raise VidLensError("VIDEO_UNAVAILABLE", "Bilibili did not return video metadata.")
            data = payload["data"]
            pages = data.get("pages") or []
            page = next((p for p in pages if int(p.get("page", 0)) == part), None)
            if page is None:
                raise VidLensError(
                    "INVALID_PART", f"Bilibili part {part} does not exist in this video."
                )
        except VidLensError as exc:
            if exc.code == "INVALID_PART":
                raise
            info = self._yt_info(url)
            if not self._matches_bilibili_part(info, part):
                # Never silently attach p1 evidence to the requested pN identity.
                raise VidLensError(
                    "PART_UNVERIFIED",
                    "Could not verify the requested Bilibili part via fallback metadata.",
                    "Retry the native platform API or import that part's subtitles.",
                ) from exc
            video = Video(
                f"bilibili:{identity}:p{part}",
                url,
                "bilibili",
                info.get("title") or identity,
                float(info.get("duration") or 0),
                self._chapters(info),
                description=str(info.get("description") or "")[:12000],
                metadata={"part": part, "metadata_source": "yt-dlp"},
            )
            return AcquiredVideo(
                video,
                self._captions_from_info(info, languages),
                [f"Native Bilibili API unavailable ({exc.code}); used yt-dlp."],
            )
        bvid, cid = str(data.get("bvid") or identity), page["cid"]
        canonical = f"https://www.bilibili.com/video/{bvid}/?p={part}"
        title = str(data.get("title") or bvid)
        if len(pages) > 1:
            title += f" · P{part} {page.get('part', '')}"
        video = Video(
            f"bilibili:{bvid}:p{part}",
            canonical,
            "bilibili",
            title,
            float(page.get("duration") or 0),
            description=str(data.get("desc") or "")[:12000],
            metadata={
                "part": part,
                "parts": len(pages),
                "cid": cid,
                "aid": data.get("aid"),
                "author": (data.get("owner") or {}).get("name", ""),
                "metadata_source": "bilibili",
            },
        )
        segments: list[Segment] = []
        try:
            player = self._json(
                "https://api.bilibili.com/x/player/wbi/v2", {"cid": cid, "bvid": bvid}
            )
            if (player.get("data") or {}).get("need_login_subtitle"):
                warnings.append(
                    "Bilibili requires login to access these subtitles. Supply an explicit cookie file or use local subtitles, frames, or explicit transcription."
                )
            tracks = ((player.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
            tracks.sort(
                key=lambda track: (
                    _language_rank(track.get("lan", ""), languages),
                    str(track.get("lan", "")).startswith("ai-"),
                )
            )
            for track in tracks[:4]:
                target = track.get("subtitle_url", "")
                if not target:
                    continue
                target = "https:" + target if target.startswith("//") else target
                language = str(track.get("lan") or "")
                source = (
                    "automatic" if language.startswith("ai-") or track.get("ai_type") else "manual"
                )
                segments = parse_subtitles(
                    self._get(target), "bilibili", source, language.removeprefix("ai-")
                )
                if segments:
                    break
        except VidLensError as exc:
            warnings.append(f"Bilibili native captions unavailable ({exc.code}).")
        if not segments:
            try:
                info = self._yt_info(canonical)
                if self._matches_bilibili_part(info, part):
                    segments = self._captions_from_info(info, languages)
                else:
                    warnings.append(
                        "Fallback captions omitted because their Bilibili part could not be verified."
                    )
            except VidLensError as exc:
                warnings.append(f"Bilibili caption fallback unavailable ({exc.code}).")
        if segments:
            video.language = segments[0].language
        return AcquiredVideo(video, segments, warnings)

    @staticmethod
    def _matches_bilibili_part(info: dict, part: int) -> bool:
        # yt-dlp's BiliBiliIE uses BV..._pN; single-part videos omit the suffix.
        match = re.search(r"_p(\d+)$", str(info.get("id", "")))
        if match:
            return int(match[1]) == part
        if info.get("playlist_index") is not None:
            return str(info["playlist_index"]) == str(part)
        return part == 1

    @staticmethod
    def _chapters(info: dict) -> list[dict]:
        chapters = []
        for row in info.get("chapters") or []:
            try:
                start, end = float(row["start_time"]), float(row["end_time"])
                if 0 <= start < end and math.isfinite(end):
                    chapters.append(
                        {"start": start, "end": end, "title": str(row.get("title") or "")}
                    )
            except (KeyError, ValueError, TypeError):
                continue
        return chapters

    def _ffmpeg(self) -> str:
        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise VidLensError(
                "FFMPEG_MISSING",
                "FFmpeg is unavailable.",
                "Install imageio-ffmpeg or configure IMAGEIO_FFMPEG_EXE.",
            ) from exc
        # yt-dlp locates its companion executables by their canonical basename.
        directory = self.cache_dir / "bin"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if target.is_symlink() and target.resolve() != Path(executable).resolve():
            target.unlink()
        if not target.exists():
            try:
                target.symlink_to(Path(executable).resolve())
            except OSError:
                return executable
        return str(target)

    def fetch_media(
        self,
        video: Video,
        destination: Path,
        start: float = 0,
        end: float | None = None,
        audio_only: bool = False,
    ) -> MediaAsset:
        if (
            not math.isfinite(start)
            or start < 0
            or (end is not None and (not math.isfinite(end) or end <= start))
        ):
            raise VidLensError(
                "INVALID_INTERVAL", "Media interval must have finite 0 <= start < end."
            )
        if video.duration > 0:
            if start >= video.duration:
                raise VidLensError(
                    "INVALID_INTERVAL", "The requested interval starts after the video ends."
                )
            end = min(end, video.duration) if end is not None else video.duration
        canonical = normalize_url(video.url)
        destination = Path(destination)
        interval_key = f"{start:.3f}-{end:.3f}" if end is not None else f"{start:.3f}-end"
        if destination.suffix:
            directory, stem = destination.parent, destination.stem
        else:
            directory = destination
            stem = ("audio-" if audio_only else "video-") + interval_key
        directory.mkdir(parents=True, exist_ok=True)
        try:
            max_height = int(os.environ.get("VIDLENS_MAX_VIDEO_HEIGHT", "1080"))
        except ValueError as exc:
            raise VidLensError(
                "VIDEO_HEIGHT", "VIDLENS_MAX_VIDEO_HEIGHT must be 720, 1080, or 2160."
            ) from exc
        if max_height not in {720, 1080, 2160}:
            raise VidLensError(
                "VIDEO_HEIGHT", "VIDLENS_MAX_VIDEO_HEIGHT must be 720, 1080, or 2160."
            )
        command = self._yt_base() + [
            "--ffmpeg-location",
            self._ffmpeg(),
            "--no-part",
            "--max-filesize",
            "1G",
            "--restrict-filenames",
            "-f",
            "bestaudio/best"
            if audio_only
            else f"bestvideo[height<={max_height}]/best[height<={max_height}]/best",
            "-o",
            str(directory / (stem + ".%(ext)s")),
            "--print",
            "after_move:filepath",
            "--no-quiet",
        ]
        if start > 0 or end is not None:
            section = f"*{start}-{end if end is not None else 'inf'}"
            command += ["--download-sections", section, "--force-keyframes-at-cuts"]
        output = self._run(command + ["--", canonical], timeout=300)
        candidates = [Path(line.strip()) for line in output.splitlines() if line.strip()]
        path = next(
            (
                p
                for p in reversed(candidates)
                if p.is_file() and p.parent.resolve() == directory.resolve()
            ),
            None,
        )
        if path is None:
            raise VidLensError(
                "MEDIA_MISSING",
                "The downloader did not produce a readable media file.",
                "Run vidlens doctor or import a local video.",
            )
        duration = max(0, end - start) if end is not None else 0
        return MediaAsset(path.resolve(), start, duration, audio_only)

    def transcribe(
        self, asset: MediaAsset, language: str | None = None, model: str = "base"
    ) -> list[Segment]:
        if not asset.path.is_file():
            raise VidLensError("MEDIA_MISSING", "The media file for transcription does not exist.")
        if not Path(model).is_dir() and not re.fullmatch(r"[A-Za-z0-9_./-]{1,160}", model):
            raise VidLensError("ASR_MODEL", "Invalid local Whisper model name or path.")
        model_cache = (self.cache_dir / "models" / "faster-whisper").resolve()
        model_cache.mkdir(parents=True, exist_ok=True)
        output = self._run(
            [
                sys.executable,
                "-m",
                "vidlens.sources",
                "whisper",
                str(asset.path.resolve()),
                model,
                language or "",
                str(model_cache),
            ],
            timeout=1200,
        )
        try:
            segments = parse_subtitles(
                output, "json", source=f"whisper:{model}", language=language or ""
            )
        except VidLensError as exc:
            raise VidLensError(
                "ASR_FAILED", "Whisper did not return valid timed segments."
            ) from exc
        bounded = []
        for segment in segments:
            if asset.duration > 0:
                segment.end = min(segment.end, asset.duration)
                if segment.start >= segment.end:
                    continue
            segment.start += asset.offset
            segment.end += asset.offset
            segment.kind = "transcript"
            bounded.append(segment)
        return bounded


def _decode_asr_audio(path: Path):
    """Use the bundled decoder instead of version-sensitive PyAV kwargs."""
    import imageio_ffmpeg
    import numpy as np

    decoded = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=300,
    )
    return np.frombuffer(decoded.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _worker() -> None:
    """Private subprocess boundary: bounded model/network operations, no shell."""
    if len(sys.argv) < 3:
        raise SystemExit(2)
    if sys.argv[1] == "youtube":
        from requests import Session
        from youtube_transcript_api import YouTubeTranscriptApi

        class TimedSession(Session):
            def request(self, *args, **kwargs):
                kwargs.setdefault("timeout", (8, 15))
                return super().request(*args, **kwargs)

        tracks = list(YouTubeTranscriptApi(http_client=TimedSession()).list(sys.argv[2]))
        requested = tuple(json.loads(sys.argv[3]))
        tracks.sort(key=lambda t: (_language_rank(t.language_code, requested), t.is_generated))
        if not tracks:
            raise SystemExit("No captions available")
        transcript = tracks[0].fetch()
        print(
            json.dumps(
                {
                    "segments": transcript.to_raw_data(),
                    "source": "automatic" if transcript.is_generated else "manual",
                    "language": transcript.language_code,
                },
                ensure_ascii=False,
            )
        )
    elif sys.argv[1] == "whisper":
        # Keep model blobs, auxiliary downloads, and Xet chunks in the same
        # explicitly selected workspace/cache root, before importing HF code.
        os.environ["HF_HOME"] = sys.argv[5]
        os.environ["HF_XET_CACHE"] = str(Path(sys.argv[5]) / "xet")
        import av

        with av.open(sys.argv[2]) as media:
            if not media.streams.audio:
                raise SystemExit("VIDLENS_NO_AUDIO")
        from faster_whisper import WhisperModel

        from .asr_models import OFFICIAL_MODELS, cached_official_model, download_official_model

        name, cache = sys.argv[3], Path(sys.argv[5])
        cached = cached_official_model(name, cache)
        try:
            try:
                model = WhisperModel(
                    str(cached) if cached else name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(cache),
                    local_files_only=True,
                )
            except Exception as cache_error:
                if type(cache_error).__name__ not in {
                    "LocalEntryNotFoundError",
                    "FileNotFoundError",
                }:
                    raise
                model = WhisperModel(
                    name, device="cpu", compute_type="int8", download_root=str(cache)
                )
        except Exception as exc:
            network_modules = {
                "httpx",
                "httpx2",
                "httpcore",
                "httpcore2",
                "huggingface_hub",
                "requests",
                "urllib3",
            }
            if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "on", "yes", "true"}:
                raise
            if (
                name not in OFFICIAL_MODELS
                or type(exc).__module__.split(".")[0] not in network_modules
            ):
                raise
            try:
                local = download_official_model(name, cache)
            except VidLensError as download_error:
                raise SystemExit("VIDLENS_ASR_MODEL_DOWNLOAD") from download_error
            model = WhisperModel(str(local), device="cpu", compute_type="int8")
        audio = _decode_asr_audio(Path(sys.argv[2]))
        segments, _info = model.transcribe(audio, language=sys.argv[4] or None, vad_filter=True)
        print(
            json.dumps(
                [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
                ensure_ascii=False,
            )
        )
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    _worker()
