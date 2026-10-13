"""Командная строка.

    python -m salon_analytics demo --out out/
    python -m salon_analytics report --ticks ticks.jsonl --persons persons.json --out out/
    python -m salon_analytics markup --config config/salon.yaml
    NVR_PASSWORD=... python -m salon_analytics probe --nvr 192.168.1.64 --out probe/
    python -m salon_analytics models --out models/
    python -m salon_analytics register --photos masters/ --out data/masters.json
    python -m salon_analytics nightly --day 2026-10-14 [--telegram]
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


def nightly(args) -> None:
    from .roles import assign_roles, load_masters
    cfg = load_config(args.config)
    day = args.day or datetime.now(report.salon_tz(cfg)).strftime("%Y-%m-%d")
    d = Path(args.data) / day
    if not (d / "ticks.jsonl").exists():
        raise SystemExit(f"нет данных за {day}: {d}/ticks.jsonl")
    ticks = [Tick(**x) for x in _read_jsonl(d / "ticks.jsonl")]
    persons = {}
    if (d / "persons.json").exists():
        for x in json.loads((d / "persons.json").read_text(encoding="utf-8")):
            persons[x["gid"]] = Person(x["gid"], reid=x.get("reid", []),
                                       created_at_entrance=x.get("created_at_entrance", True))
    for t in ticks:
        persons.setdefault(t.gid, Person(t.gid))
    faces: dict[str, list[FaceCandidate]] = {}
    if (d / "faces.jsonl").exists():
        for x in _read_jsonl(d / "faces.jsonl"):
            faces.setdefault(x["gid"], []).append(FaceCandidate(**x))
    outages = [Outage(**x) for x in _read_jsonl(d / "outages.jsonl")] if (d / "outages.jsonl").exists() else []
    masters_file = Path(args.masters) if args.masters else Path(args.data) / "masters.json"
    masters = load_masters(masters_file) if masters_file.exists() else {}
    if not masters:
        print("внимание: мастера не зарегистрированы (register) — все будут клиентами или «сотрудник?»")
    assign_roles(persons, faces, ticks, masters, cfg.params)
    r = analyze_day(DayInput(ticks, persons, faces, outages, crm_checks=args.crm_checks), cfg)
    out = d / "report"
    write_outputs(r, cfg, out, str(Path(args.data) / "salon.db"), photos_base=d)
    text = report.telegram_summary(r, cfg)
    print(text)
    print(f"отчёт: {out / 'masters.html'}")
    if args.telegram:
        from .telegram import send_report
        send_report(text, out / "masters.html")
    removed = cleanup_old_days(Path(args.data), day, cfg.params.retention_days)
    if removed:
        print(f"удалены данные старше {cfg.params.retention_days} дней: {', '.join(removed)}")


def cleanup_old_days(data: Path, today: str, keep_days: int) -> list[str]:
    """Фото лиц и треки хранятся не дольше срока из ТЗ (по умолчанию 90 дней)."""
    import shutil
    from datetime import date, timedelta
    limit = date.fromisoformat(today) - timedelta(days=keep_days)
    removed = []
    for d in sorted(data.iterdir()) if data.exists() else []:
        try:
            if d.is_dir() and date.fromisoformat(d.name) < limit:
                shutil.rmtree(d)
                removed.append(d.name)
        except ValueError:
            continue
    return removed


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
    mo = sub.add_parser("models", help="скачать модели YOLO, YuNet, SFace (OSNet BoxMOT скачает сам)")
    mo.add_argument("--out", default="models")
    mo.add_argument("--coreml", action="store_true", help="экспорт YOLO в CoreML (только на Mac)")
    rg = sub.add_parser("register", help="зарегистрировать мастеров по фото masters/<Имя>/*.jpg")
    rg.add_argument("--photos", default="masters")
    rg.add_argument("--models", default="models")
    rg.add_argument("--out", default="data/masters.json")
    nt = sub.add_parser("nightly", help="вечерний отчёт по файлам дня из data/")
    nt.add_argument("--config", default=str(DEFAULT_CONFIG))
    nt.add_argument("--data", default="data")
    nt.add_argument("--day", default=None, help="ГГГГ-ММ-ДД, по умолчанию сегодня")
    nt.add_argument("--masters", default=None, help="по умолчанию data/masters.json")
    nt.add_argument("--crm-checks", type=int, default=None)
    nt.add_argument("--telegram", action="store_true", help="отправить сводку и masters.html в Telegram")
    args = ap.parse_args(argv)
    if args.cmd == "models":
        from .pipeline.faces import download_models
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        for p in download_models(out):
            print("ok", p)
        from ultralytics import YOLO
        for w in ("yolo11n-pose.pt", "yolo11n.pt"):
            m = YOLO(str(out / w) if (out / w).exists() else w)
            if not (out / w).exists():
                Path(w).replace(out / w)
            print("ok", out / w)
            if args.coreml:
                print("ok", m.export(format="coreml", nms=False))
        return
    if args.cmd == "register":
        from .roles import register_masters
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        counts = register_masters(args.photos, args.models, args.out)
        for name, n in counts.items():
            print(f"{name}: {n} фото с лицом" + ("" if n else "  ← НЕТ ЛИЦА, добавьте фото анфас"))
        raise SystemExit(0 if counts and all(counts.values()) else 1)
    if args.cmd == "nightly":
        nightly(args)
        return
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
