"""Главный экран: мастер (фото с камеры) → число клиентов → фото каждого клиента.

Один самодостаточный HTML-файл: картинки встраиваются внутрь, поэтому его
можно переслать в Telegram или открыть на любом устройстве.
"""
from __future__ import annotations

import base64
import html
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import SalonConfig
from .engine import DayResult
from .models import FaceCandidate, PhotoChoice, Visit
from .report import plural, salon_tz

CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--muted:#6e6e73;--line:#e3e1dc;--warn:#b26a00;--warn-bg:#fff4e0;--accent:#2f5d50}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1f1f1f;--ink:#f2f2f2;--muted:#a1a1a6;--line:#333;--warn:#ffb340;--warn-bg:#3a2a10;--accent:#8fc9b5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.4 -apple-system,system-ui,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 24px}
.master{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:20px}
.head{display:flex;align-items:center;gap:16px;margin-bottom:16px}
.mphoto{width:88px;height:88px;border-radius:50%;object-fit:cover;flex:none;border:3px solid var(--accent);background:var(--line)}
.name{font-size:20px;font-weight:600}.count{font-size:34px;font-weight:700;color:var(--accent);line-height:1.1}
.note{color:var(--muted);font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:12px}
.client{text-align:center;font-size:13px;color:var(--muted)}
.client img,.client .ph{width:100%;aspect-ratio:1;border-radius:12px;object-fit:cover;display:block;background:var(--line);border:2px solid transparent}
.client .ph{display:flex;align-items:center;justify-content:center;font-size:12px;color:var(--muted)}
.client.check img,.client.check .ph{border-color:var(--warn)}
.client .t{margin-top:4px;color:var(--ink)}.badge{display:inline-block;margin-top:2px;padding:0 6px;border-radius:6px;background:var(--warn-bg);color:var(--warn);font-size:11px}
.legend{color:var(--muted);font-size:13px;margin-top:8px}
"""


def _img_src(c: Optional[FaceCandidate], base: Path) -> Optional[str]:
    if c is None or not c.path:
        return None
    path = Path(c.path)
    if not path.is_absolute():
        path = base / path
    if not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _photo(choice: Optional[PhotoChoice], base: Path, cls: str, alt: str) -> str:
    src = _img_src(choice.best if choice else None, base)
    if src:
        return f'<img class="{cls}" src="{src}" alt="{html.escape(alt)}">'
    if cls == "mphoto":
        return f'<div class="mphoto" role="img" aria-label="{html.escape(alt)}"></div>'
    return '<div class="ph">нет фото</div>'


def _client_tile(v: Visit, cfg: SalonConfig, base: Path) -> str:
    tz = salon_tz(cfg)
    t = datetime.fromtimestamp(v.start, tz).strftime("%H:%M")
    check = v.status == "на проверке"
    badge = '<div class="badge">проверить</div>' if check else ""
    face = "" if not v.photo or v.photo.has_face else " (фигура)"
    return (f'<div class="client{" check" if check else ""}">{_photo(v.photo, base, "", f"клиент {t}")}'
            f'<div class="t">{t}{face}</div>{badge}</div>')


def render(day: DayResult, cfg: SalonConfig, photos_base: str | Path = ".") -> str:
    base = Path(photos_base)
    tz = salon_tz(cfg)
    visits = sorted(day.visits, key=lambda v: v.start)
    date = datetime.fromtimestamp(visits[0].start, tz).strftime("%d.%m.%Y") if visits else ""
    by_master: dict[str, list[Visit]] = {}
    for v in visits:
        if not v.staff_client:
            by_master.setdefault(v.master or "", []).append(v)
    names = sorted(set(day.masters) | {m for m in by_master if m},
                   key=lambda n: (-len(by_master.get(n, [])), n))
    total = sum(len(vs) for m, vs in by_master.items() if m)

    blocks = []
    for name in names:
        vs = by_master.get(name, [])
        unsure = sum(1 for v in vs if v.status == "на проверке")
        d = day.masters.get(name)
        notes = []
        if unsure:
            notes.append(f"из них {unsure} на проверке")
        if d and d.colleagues:
            notes.append(f"ещё стриг(ла) коллег: {d.colleagues}")
        tiles = "".join(_client_tile(v, cfg, base) for v in vs) or '<div class="note">клиентов нет</div>'
        blocks.append(
            f'<section class="master"><div class="head">'
            f'{_photo(day.master_photos.get(name), base, "mphoto", name)}'
            f'<div><div class="name">{html.escape(name)}</div>'
            f'<div class="count">{plural(len(vs), "клиент", "клиента", "клиентов")}</div>'
            f'<div class="note">{"; ".join(notes)}</div></div></div>'
            f'<div class="grid">{tiles}</div></section>')
    if by_master.get(""):
        tiles = "".join(_client_tile(v, cfg, base) for v in by_master[""])
        blocks.append(f'<section class="master"><div class="name">Мастер не определён</div>'
                      f'<div class="note" style="margin-bottom:12px">нужна проверка: '
                      f'{len(by_master[""])}</div><div class="grid">{tiles}</div></section>')

    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Мастера и клиенты {date}</title><style>{CSS}</style></head><body><main>'
            f'<h1>Мастера и клиенты</h1><p class="sub">{html.escape(cfg.name)} · {date} · '
            f'всего {plural(total, "клиент", "клиента", "клиентов")}</p>'
            f'{"".join(blocks)}'
            f'<p class="legend">Фото мастера и фото клиентов — кадры с камер. Кадр, похожий на лицо '
            f'другого человека, в отчёт не попадает. Оранжевая рамка — визит на проверке.</p>'
            f'</main></body></html>')
