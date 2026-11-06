"""An exported explanation must keep its source and image attribution coherent."""

import pytest
from PIL import Image

from vidlens import learning
from vidlens.models import Frame, Segment, Video, VidLensError
from vidlens.service import VideoService


def seed(tmp_path):
    service = VideoService(tmp_path / "library")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"original local media")
    picture = tmp_path / "original.jpg"
    Image.new("RGB", (20, 10), "red").save(picture)
    video = Video("lesson", "https://example.org/lesson", "local", "Original lesson", 10,
                  metadata={"revision": "original", "local_media_path": str(media)})
    cue = Segment(0, 10, "The diagram is red.")
    service.store.save(video, [cue], [])
    plan = service.study_status(video.id)
    service.store.save_frames(video.id, [Frame(1, picture, 20, 10)])
    service.save_note(video.id, plan["sections"][0]["id"], plan["revision"], {
        "title": "The diagram", "summary": "An explanation of the red diagram.",
        "claims": [{"text": "The frame is red.", "evidence_ids": [cue.id, "frame:1"]}],
    })
    return service, video, cue, media, picture


@pytest.mark.parametrize("change", ["metadata", "transcript", "local_media"])
def test_export_rejects_source_changes_after_reading_notes(tmp_path, monkeypatch, change):
    service, video, cue, media, _ = seed(tmp_path)
    original_link = learning.source_link
    changed = False

    def change_during_rendering(source, start):
        nonlocal changed
        if not changed:
            changed = True
            if change == "local_media":
                media.write_bytes(b"replacement local media with a different size")
            else:
                if change == "metadata":
                    video.metadata["revision"] = "replacement"
                    video.title = "Replacement lesson"
                else:
                    cue.text = "The diagram is blue."
                service.store.save(video, [cue], [], invalidate_derived=True)
                new_picture = tmp_path / "replacement.jpg"
                Image.new("RGB", (20, 10), "blue").save(new_picture)
                service.store.save_frames(video.id, [Frame(1, new_picture, 20, 10)])
        return original_link(source, start)

    monkeypatch.setattr(learning, "source_link", change_during_rendering)
    with pytest.raises(VidLensError) as error:
        service.export_study(video.id)
    assert changed
    assert error.value.code == "stale_revision"


def test_export_keeps_snapshot_picture_when_same_source_gets_a_new_capture(tmp_path, monkeypatch):
    service, video, _, _, picture = seed(tmp_path)
    better_picture = tmp_path / "higher-resolution.jpg"
    Image.new("RGB", (200, 100), "red").save(better_picture)
    original_link = learning.source_link
    replaced = False

    def improve_capture_during_rendering(source, start):
        nonlocal replaced
        if not replaced:
            replaced = True
            service.store.save_frames(video.id, [Frame(1, better_picture, 200, 100)])
        return original_link(source, start)

    monkeypatch.setattr(learning, "source_link", improve_capture_during_rendering)
    exported = service.export_study(video.id)
    assert exported["image_count"] == 1
    assert f"(<{picture}>)" in exported["markdown"]
    assert str(better_picture) not in exported["markdown"]
    assert service.store.frame_at(video.id, 1).path == better_picture
