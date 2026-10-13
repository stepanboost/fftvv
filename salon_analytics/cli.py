"""Командная строка.

    python -m salon_analytics demo --out out/
    python -m salon_analytics report --ticks ticks.jsonl --persons persons.json --out out/
    python -m salon_analytics markup --config config/salon.yaml
    NVR_PASSWORD=... python -m salon_analytics probe --nvr 192.168.1.64 --out probe/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from . import markup, masters_page, report, storage
from .config import default_config, load_config
from .engine import DayInput, analyze_day
from .models import FaceCandidate, Outage, Person, Role, Tick
from .sim import crowded_day, demo_day, write_demo_photos

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "salon.yaml"


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_input(args) -> DayInput:
    ticks = [Tick(**d) for d in _read_jsonl(args.ticks)]
    persons = {}
    for d in json.loads(Path(args.persons).read_text(encoding="utf-8")):
        d["role"] = Role(d.get("role", "client"))
        persons[d["gid"]] = Person(**d)
    faces: dict[str, list[FaceCandidate]] = {}
    if args.faces:
        for d in _read_jsonl(args.faces):
            faces.setdefault(d["gid"], []).append(FaceCandidate(**d))
    outages = [Outage(**d) for d in _read_jsonl(args.outages)] if args.outages else []
    return DayInput(ticks, persons, faces, outages, crm_checks=args.crm_checks)


def write_outputs(r, cfg, out: Path, db: str | None, photos_base: str | Path = ".") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "visits.csv").write_text(report.to_csv(r, cfg), encoding="utf-8-sig")
    payload = {
        "header": report.day_header(r),
        "masters": report.master_summary(r, cfg),
        "visits": report.rows(r, cfg),
        "review_queue": report.review_cards(r, cfg),
        "violations": [asdict(v) for v in r.violations],
    }
    (out / "day.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "telegram.txt").write_text(report.telegram_summary(r, cfg), encoding="utf-8")
    (out / "masters.html").write_text(masters_page.render(r, cfg, photos_base), encoding="utf-8")
    if db and r.visits:
        day = datetime.fromtimestamp(r.visits[0].start, report.salon_tz(cfg)).strftime("%Y-%m-%d")
        storage.save_day(storage.connect(db), day, r, cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="salon_analytics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None)
        sp.add_argument("--out", default="out")
        sp.add_argument("--db", default=None)
        sp.add_argument("--crm-checks", type=int, default=None)
        if name == "demo":
            sp.add_argument("--crowded", action="store_true", help="зал битком: 3+ клиента в кадре")
        if name == "report":
            sp.add_argument("--ticks", required=True)
            sp.add_argument("--persons", required=True)
            sp.add_argument("--faces")
            sp.add_argument("--outages")
            sp.add_argument("--photos", default=".", help="папка, от которой считаются пути фото")
    mp = sub.add_parser("markup", help="проверка разметки камер и реестра кресел")
    mp.add_argument("--config", default=str(DEFAULT_CONFIG))
    pp = sub.add_parser("probe", help="подключиться к регистратору и проверить все камеры")
    pp.add_argument("--config", default=str(DEFAULT_CONFIG))
    pp.add_argument("--nvr", help="IP регистратора в сети салона")
    pp.add_argument("--user", default=None, help="логин (или NVR_USER), пароль — NVR_PASSWORD или ввод")
    pp.add_argument("--main", action="store_true", help="основной поток вместо суб-потока")
    pp.add_argument("--source", action="append", default=[], metavar="CAM=URL",
                    help="подменить источник камеры (файл/URL), можно несколько раз")
    pp.add_argument("--out", default="probe")
    pp.add_argument("--timeout", type=float, default=15)
    args = ap.parse_args(argv)
    if args.cmd == "probe":
        import getpass
        import os

        from .pipeline.probe import run_probe
        sources = dict(s.split("=", 1) for s in args.source) or None
        if sources is None and not args.nvr:
            ap.error("нужен --nvr или --source")
        user = args.user or os.environ.get("NVR_USER", "admin")
        password = os.environ.get("NVR_PASSWORD")
        if sources is None and password is None:
            password = getpass.getpass(f"пароль {user}@{args.nvr}: ")
        res = run_probe(load_config(args.config), args.nvr or "", Path(args.out), user, password or "",
                        sub=not args.main, sources=sources, timeout=args.timeout)
        raise SystemExit(0 if res and all(r.ok for r in res) else 1)
    if args.cmd == "markup":
        issues = markup.validate(load_config(args.config))
        for i in issues:
            print(f"{'ОШИБКА' if i.level == 'error' else 'внимание'}: {i.where}: {i.text}")
        raise SystemExit(1 if any(i.level == "error" for i in issues) else 0)
    cfg = load_config(args.config) if args.config else default_config()
    if args.cmd == "demo":
        if not args.config:
            cfg = load_config(DEFAULT_CONFIG)
        b = crowded_day() if args.crowded else demo_day()
        inp = DayInput(b.ticks, b.persons, b.faces, crm_checks=args.crm_checks)
        write_demo_photos(b, Path(args.out))
        args.photos = args.out
    else:
        inp = load_input(args)
    r = analyze_day(inp, cfg)
    write_outputs(r, cfg, Path(args.out), args.db, getattr(args, "photos", "."))
    print(report.telegram_summary(r, cfg))


if __name__ == "__main__":
    main()
