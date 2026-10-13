"""Кто мастер, кто клиент — по готовым методам из розницы.

1. Сверка лица с зарегистрированными мастерами (как «галерея сотрудников» в
   системах исключения персонала): ID становится мастером, только если на
   нескольких кадрах лицо совпало с одним и тем же мастером, и с ним — заметно
   лучше, чем с другими.
2. Поведение: кто провёл в салоне много часов, но лицом не опознан — это
   «сотрудник?», а не клиент (стажёр, подмена). Такие визиты идут на проверку.
3. Остальные — клиенты. В их галерею лица не попадают кадры, похожие на мастеров.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .config import Params
from .models import FaceCandidate, Person, Role, Tick
from .similarity import cosine


def load_masters(path: str | Path) -> dict[str, list[list[float]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {m["name"]: m["embeddings"] for m in data}


def register_masters(photos_dir: str | Path, models_dir: str | Path, out: str | Path) -> dict[str, int]:
    """masters/<Имя>/*.jpg → masters.json. 3–5 фото анфас на мастера достаточно."""
    from .pipeline.faces import FaceEngine

    eng = FaceEngine(models_dir)
    result, counts = [], {}
    for d in sorted(p for p in Path(photos_dir).iterdir() if p.is_dir()):
        embs = [e for f in sorted(d.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                if (e := eng.embed_image(f)) is not None]
        counts[d.name] = len(embs)
        if embs:
            result.append({"name": d.name, "embeddings": embs})
    Path(out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return counts


def presence_s(ticks: Iterable[Tick]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for t in ticks:
        if t.present:
            out[t.gid] += t.dt
    return out


def assign_roles(persons: dict[str, Person], faces: dict[str, list[FaceCandidate]],
                 ticks: list[Tick], masters: dict[str, list[list[float]]], p: Params) -> dict[str, Person]:
    stay = presence_s(ticks)
    all_master = [e for embs in masters.values() for e in embs]
    for gid, person in persons.items():
        votes: dict[str, int] = defaultdict(int)
        best_sim: dict[str, float] = defaultdict(float)
        cands = [c for c in faces.get(gid, []) if c.embedding]
        for c in cands:
            sims = {name: max(cosine(c.embedding, e) for e in embs) for name, embs in masters.items() if embs}
            if not sims:
                continue
            ranked = sorted(sims.items(), key=lambda kv: -kv[1])
            name, sim = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            if sim >= p.face_same_cos and sim - second >= p.photo_own_margin:
                votes[name] += 1
                best_sim[name] = max(best_sim[name], sim)
        if votes:
            name = max(votes, key=lambda n: (votes[n], best_sim[n]))
            others = sum(v for n, v in votes.items() if n != name)
            if (votes[name] >= p.master_min_face_frames or best_sim[name] >= p.repeat_face_cos) \
                    and votes[name] > others:
                person.role, person.name = Role.MASTER, name
                person.face = list(masters[name])          # доверяем только эталону
                continue
        if stay.get(gid, 0.0) >= p.staff_presence_s:
            person.role = Role.STAFF_UNKNOWN
            continue
        person.role = Role.CLIENT
        person.face = [c.embedding for c in cands
                       if not all_master or max(cosine(c.embedding, e) for e in all_master) < p.face_same_cos]
    return persons
