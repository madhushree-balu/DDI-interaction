"""
polyguard/scripts/run_analysis.py
==================================
Command-line runner for the PolyGuard analysis pipeline.

Usage::

    python -m polyguard.scripts.run_analysis \\
        --brands "Augmentin 625 Duo Tablet" "Ascoril LS Syrup" \\
        --age 72 --gender Female \\
        --conditions Hypertension "Diabetes Type 2" "Atrial Fibrillation" \\
        --lab eGFR=42 ALT=85 platelet_count=110 INR=3.2 blood_glucose=195 \\
        --save report.json

    python -m polyguard.scripts.run_analysis --search Aug
    python -m polyguard.scripts.run_analysis --ingredients "Augmentin 625 Duo Tablet"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polyguard.core.analyser import PolyGuardAnalyser
from polyguard.core.data_loader import DataLoader


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PolyGuard — Drug Interaction Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--brands",       nargs="+", metavar="BRAND",  help="Brand names to analyse")
    mode.add_argument("--search",       metavar="PREFIX",            help="Search brand names by prefix")
    mode.add_argument("--ingredients",  metavar="BRAND",             help="List ingredients for one brand")

    p.add_argument("--data-dir", default="./datasets", help="Path to datasets directory")
    p.add_argument("--age",       type=int, help="Patient age")
    p.add_argument("--gender",    help="Patient gender")
    p.add_argument("--conditions", nargs="*", default=[], help="Patient conditions")
    p.add_argument("--lab",        nargs="*", default=[], metavar="KEY=VALUE",
                   help="Lab values e.g. eGFR=42 ALT=85")
    p.add_argument("--no-xai",     action="store_true", help="Disable XAI explanations")
    p.add_argument("--save",       metavar="FILE", help="Save JSON report to FILE")
    p.add_argument("--limit",      type=int, default=10, help="Limit for brand search results")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    loader   = DataLoader(args.data_dir).load()
    analyser = PolyGuardAnalyser(loader)

    # ── Brand search mode ─────────────────────────────────────────────────────
    if args.search:
        result = analyser.search_brands(args.search, limit=args.limit)
        print(f"\nBrand search: '{args.search}*'")
        print(f"Found {result.total_found} result(s):")
        for r in result.results:
            print(f"  • {r}")
        return 0

    # ── Ingredient lookup mode ────────────────────────────────────────────────
    if args.ingredients:
        result = analyser.get_ingredients(args.ingredients)
        print(f"\nIngredients for: {args.ingredients}")
        if result.found:
            for ing in result.ingredients:
                print(f"  • {ing}")
        else:
            print("  (brand not found)")
        return 0

    # ── Full analysis mode ────────────────────────────────────────────────────
    patient_data = None
    if args.age or args.conditions or args.lab:
        lab_values: dict = {}
        for kv in args.lab:
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    lab_values[k.strip()] = float(v.strip())
                except ValueError:
                    print(f"Warning: could not parse lab value '{kv}' — skipping.", file=sys.stderr)
        patient_data = {
            "age":        args.age,
            "gender":     args.gender,
            "conditions": args.conditions,
            "lab_values": lab_values,
        }

    result = analyser.analyse(
        brand_names  = args.brands,
        patient_data = patient_data,
        explain      = not args.no_xai,
    )

    _print_result(result.model_dump())

    if args.save:
        path = Path(args.save)
        path.write_text(
            json.dumps(result.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n💾 Report saved → {path}")

    return 0


def _print_result(d: dict) -> None:
    W = 72
    status = d.get("status", "")
    print(f"\n{'='*W}")
    print(f"  POLYGUARD ANALYSIS RESULT")
    print(f"{'='*W}")
    print(f"  Status: {status}")

    if status == "NO_INTERACTIONS":
        print(f"  {d.get('message','')}")
        return

    summary = d.get("summary") or {}
    print(f"  Risk Level  : {summary.get('overall_risk_level','')} {summary.get('risk_color','')}")
    print(f"  Action      : {summary.get('primary_action','')}")
    print(f"  Total Score : {d.get('total_score',0)}")
    print(f"  Interactions: {len(d.get('interactions_found',[]))}")
    print(f"  Organs      : {d.get('num_organs_affected',0)}")
    print(f"  Cascades    : {len(d.get('cascades',[]))}")

    print(f"\n  {'─'*W}")
    print(f"  SEVERITY BREAKDOWN")
    for b in d.get("severity_breakdown", []):
        print(f"  {b['icon']}  {b['drugs']}  [{b['severity']}]  score={b['score']}")
        desc = b.get("description", "")
        if desc and desc != "No description available":
            print(f"       {desc[:100]}{'…' if len(desc) > 100 else ''}")

    organs = d.get("organ_systems", [])
    if organs:
        print(f"\n  {'─'*W}")
        print(f"  ORGAN SYSTEMS")
        for s in organs[:8]:
            score = s.get("adjusted_score") or s.get("score", 0)
            bar   = "█" * min(int(score / 4), 20)
            print(f"  {s['icon']}  {s['system']:<28} {score:>4}  {bar}")

    cascades = d.get("cascades", [])
    if cascades:
        print(f"\n  {'─'*W}")
        print(f"  CASCADE ALERTS")
        for c in cascades:
            print(f"  🔗  {c['organ_system']}  [{c['alert_level']}]  score={c['cumulative_score']}")

    print(f"\n{'='*W}\n")


if __name__ == "__main__":
    sys.exit(main())