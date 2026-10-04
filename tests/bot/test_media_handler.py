from types import SimpleNamespace

from bot.handlers.media import MAX_PHOTO_BYTES, _select_photo


def test_select_photo_uses_largest_variant_under_two_mb():
    small = SimpleNamespace(width=320, height=200, file_size=50_000)
    accepted = SimpleNamespace(width=1280, height=720, file_size=1_900_000)
    too_large = SimpleNamespace(
        width=1920,
        height=1080,
        file_size=MAX_PHOTO_BYTES + 1,
    )

    assert _select_photo([small, accepted, too_large]) is accepted


def test_select_photo_returns_none_when_all_variants_are_too_large():
    photo = SimpleNamespace(width=1920, height=1080, file_size=MAX_PHOTO_BYTES + 1)
    assert _select_photo([photo]) is None
