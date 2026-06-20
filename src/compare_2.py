#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_2.py
Usage:
  python compare_2.py --run_a ./green_results_2/green_YYYYMMDD_HHMMSS \
                      --run_b ./baseline_training_results/baseline_YYYYMMDD_HHMMSS \
                      --outdir ./reports_2

Generates:
- comparison_chart.png
- loss_curve.png (if trainer_state.json contains loss history)
- Comparison_Report.pdf

Adds GreenTrainer2-specific rows to the PDF table:
  - Skipped steps, Skip rate (%), Early accum exits, Micro-batches saved
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import black, whitesmoke


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_latest_run_report(run_dir: str) -> str:
    # Prefer *_run_report_*.json (unified report from fine-tune_2.py) over any
    # other *_report_*.json files (e.g. green_training_detail saved by GreenTrainer2).
    run_reports = glob.glob(os.path.join(run_dir, "*_run_report_*.json"))
    if run_reports:
        return max(run_reports, key=os.path.getmtime)
    files = glob.glob(os.path.join(run_dir, "*_report_*.json"))
    if not files:
        raise FileNotFoundError(f"No *_report_*.json found in {run_dir}")
    return max(files, key=os.path.getmtime)


def find_trainer_state(run_dir: str) -> Optional[str]:
    direct = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(direct):
        return direct
    candidates = glob.glob(os.path.join(run_dir, "**", "trainer_state.json"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_loss_from_trainer_state(path: str) -> Optional[Tuple[List[int], List[float]]]:
    try:
        st = load_json(path)
        hist = st.get("log_history", [])
        steps, losses = [], []
        for it in hist:
            if "loss" in it and "step" in it:
                steps.append(int(it["step"]))
                losses.append(float(it["loss"]))
        if len(steps) < 2:
            return None
        pairs = sorted(zip(steps, losses), key=lambda x: x[0])
        s, l = zip(*pairs)
        return list(s), list(l)
    except Exception:
        return None


def pct_change(new: float, old: float) -> Optional[float]:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def impact_text(val_a: float, val_b: float) -> str:
    pct = pct_change(val_b, val_a)
    if pct is None:
        return "N/A"
    pct = -pct
    if pct > 0:
        return f"+{pct:.1f}%"
    elif pct < 0:
        return f"{pct:.1f}%"
    return "0.0%"


def fmt_wh(wh: float) -> str:
    return f"{wh:.4f} Wh"


def fmt_s(s: float) -> str:
    return f"{s:.1f} s"


def fmt_kg(kg: float) -> str:
    if kg == 0:
        return "0"
    g = kg * 1000
    mg = g * 1000
    if mg < 1000:
        return f"{mg:.2f} mg"
    if g < 1000:
        return f"{g:.2f} g"
    return f"{kg:.6f} kg"


def fmt_int(v: Any) -> str:
    try:
        return str(int(v))
    except Exception:
        return "0"


def fmt_pct(v: Any) -> str:
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return "0.00%"


@dataclass
class Run:
    label: str
    dir: str
    report_path: str
    report: Dict[str, Any]
    trainer_state_path: Optional[str]


def load_run(run_dir: str, label: str) -> Run:
    rp = find_latest_run_report(run_dir)
    r = load_json(rp)
    tsp = r.get("trainer_state_path") or find_trainer_state(run_dir)
    return Run(label=label, dir=run_dir, report_path=rp, report=r, trainer_state_path=tsp)


def generate_charts(a: Run, b: Run, outdir: str) -> Tuple[str, str, str, Optional[str]]:
    ensure_dir(outdir)
    fig_energy = os.path.join(outdir, "energy_chart.png")
    fig_time = os.path.join(outdir, "time_chart.png")
    fig_co2 = os.path.join(outdir, "co2_chart.png")
    fig_loss = os.path.join(outdir, "loss_curve.png")

    a_energy = float(a.report.get("total_energy_wh", 0.0) or 0.0)
    b_energy = float(b.report.get("total_energy_wh", 0.0) or 0.0)
    a_time = float(a.report.get("time_seconds_wall", 0.0) or 0.0)
    b_time = float(b.report.get("time_seconds_wall", 0.0) or 0.0)
    a_co2 = float(a.report.get("co2_kg_operational_x_pue", 0.0) or 0.0)
    b_co2 = float(b.report.get("co2_kg_operational_x_pue", 0.0) or 0.0)

    labels = [a.label, b.label]
    x = np.arange(len(labels))
    width = 0.5

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    r1 = ax1.bar(x, [a_energy, b_energy], width, label="Energy (Wh)", color="tab:blue")
    r1[0].set_hatch("/")
    if len(r1) > 1:
        r1[1].set_hatch("x")
    ax1.set_title("Total Energy (Wh)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.bar_label(r1, padding=3, fmt="%.4f")
    plt.tight_layout()
    plt.savefig(fig_energy)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    r2 = ax2.bar(x, [a_time, b_time], width, label="Time (s)", color="tab:orange")
    r2[0].set_hatch("\\\\")
    if len(r2) > 1:
        r2[1].set_hatch(".")
    ax2.set_title("Training Time (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.bar_label(r2, padding=3, fmt="%.1f")
    plt.tight_layout()
    plt.savefig(fig_time)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    r3 = ax3.bar(x, [a_co2, b_co2], width, label="CO2 (kg)", color="tab:green")
    r3[0].set_hatch("/")
    if len(r3) > 1:
        r3[1].set_hatch("x")
    ax3.set_title("CO2 Emissions (kg, operational×PUE)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.bar_label(r3, padding=3, fmt="%.6f")
    plt.tight_layout()
    plt.savefig(fig_co2)
    plt.close(fig3)

    a_loss = load_loss_from_trainer_state(a.trainer_state_path) if a.trainer_state_path else None
    b_loss = load_loss_from_trainer_state(b.trainer_state_path) if b.trainer_state_path else None

    if a_loss or b_loss:
        plt.figure(figsize=(10, 6))
        if a_loss:
            s, l = a_loss
            plt.plot(s, l, marker="o", label=f"{a.label} loss")
        if b_loss:
            s, l = b_loss
            plt.plot(s, l, marker="x", linestyle="--", label=f"{b.label} loss")
        plt.title("Training loss (trainer_state.json)")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_loss)
        plt.close()
        return fig_energy, fig_time, fig_co2, fig_loss

    if os.path.exists(fig_loss):
        os.remove(fig_loss)
    return fig_energy, fig_time, fig_co2, None


def create_pdf(
    a: Run,
    b: Run,
    fig_energy: str,
    fig_time: str,
    fig_co2: str,
    fig_loss: Optional[str],
    outdir: str,
) -> str:
    ensure_dir(outdir)
    pdf_path = os.path.join(outdir, "Comparison_Report.pdf")

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    body.spaceAfter = 8

    def get(d: Dict[str, Any], k: str) -> float:
        return float(d.get(k, 0.0) or 0.0)

    aE, bE = get(a.report, "total_energy_wh"), get(b.report, "total_energy_wh")
    aE_gpu, bE_gpu = get(a.report, "gpu_energy_wh"), get(b.report, "gpu_energy_wh")
    aE_cpu, bE_cpu = get(a.report, "cpu_energy_wh"), get(b.report, "cpu_energy_wh")
    aP_gpu, bP_gpu = get(a.report, "gpu_power_w"), get(b.report, "gpu_power_w")
    aP_cpu, bP_cpu = get(a.report, "cpu_power_w"), get(b.report, "cpu_power_w")
    aT, bT = get(a.report, "time_seconds_wall"), get(b.report, "time_seconds_wall")
    aC, bC = get(a.report, "co2_kg_operational_x_pue"), get(b.report, "co2_kg_operational_x_pue")
    aEmb, bEmb = get(a.report, "co2_kg_embodied_amortized"), get(b.report, "co2_kg_embodied_amortized")
    aTot, bTot = get(a.report, "co2_kg_total_pue_plus_embodied"), get(b.report, "co2_kg_total_pue_plus_embodied")

    # GreenTrainer2 metrics (default 0 for runs that don't have them)
    a_skipped_steps = a.report.get("skipped_steps", 0)
    b_skipped_steps = b.report.get("skipped_steps", 0)
    a_skip_rate = a.report.get("skip_rate_pct", 0.0)
    b_skip_rate = b.report.get("skip_rate_pct", 0.0)
    a_early_exits = a.report.get("early_accum_exits", 0)
    b_early_exits = b.report.get("early_accum_exits", 0)
    a_mb_saved = a.report.get("micro_batches_saved", 0)
    b_mb_saved = b.report.get("micro_batches_saved", 0)

    energy_imp = impact_text(aE, bE)
    energy_gpu_imp = impact_text(aE_gpu, bE_gpu)
    energy_cpu_imp = impact_text(aE_cpu, bE_cpu)
    power_gpu_imp = impact_text(aP_gpu, bP_gpu)
    power_cpu_imp = impact_text(aP_cpu, bP_cpu)
    time_imp = impact_text(aT, bT)
    co2_imp = impact_text(aC, bC)
    total_imp = impact_text(aTot, bTot)
    embodied_imp = impact_text(aEmb, bEmb)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    content = []

    content.append(Paragraph("Fine-tuning Comparison Report (GreenTrainer2)", title_style))
    content.append(Paragraph(f"Generated: {now_str()}", body))
    content.append(Spacer(1, 10))

    content.append(Paragraph("1. Inputs", h1))
    content.append(Paragraph(f"{a.label}: {a.report_path}", body))
    content.append(Paragraph(f"{b.label}: {b.report_path}", body))
    content.append(Spacer(1, 8))

    content.append(Paragraph("2. Disaggregated Results", h1))
    data = [
        ["Metric", a.label, b.label, "Impact (A vs B)"],
        ["Total Energy", fmt_wh(aE), fmt_wh(bE), energy_imp],
        ["GPU Energy", fmt_wh(aE_gpu), fmt_wh(bE_gpu), energy_gpu_imp],
        ["CPU/SoC Energy", fmt_wh(aE_cpu), fmt_wh(bE_cpu), energy_cpu_imp],
        ["Avg GPU Power (W)", f"{aP_gpu:.2f}", f"{bP_gpu:.2f}", power_gpu_imp],
        ["Avg CPU/SoC Power (W)", f"{aP_cpu:.2f}", f"{bP_cpu:.2f}", power_cpu_imp],
        ["Time (wall)", fmt_s(aT), fmt_s(bT), time_imp],
        ["CO2 (operational×PUE)", fmt_kg(aC), fmt_kg(bC), co2_imp],
        ["CO2 (embodied amortized)", fmt_kg(aEmb), fmt_kg(bEmb), embodied_imp],
        ["CO2 total (PUE+embodied)", fmt_kg(aTot), fmt_kg(bTot), total_imp],
    ]

    table = Table(data, colWidths=[150, 120, 120, 140])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("GRID", (0, 0), (-1, -1), 1, black),
            ]
        )
    )
    content.append(table)
    content.append(Spacer(1, 14))

    content.append(Paragraph("3. GreenTrainer2 Efficiency Metrics", h1))
    data2 = [
        ["Metric", a.label, b.label, "Impact (A vs B)"],
        [
            "Skipped steps",
            fmt_int(a_skipped_steps),
            fmt_int(b_skipped_steps),
            impact_text(float(a_skipped_steps), float(b_skipped_steps)),
        ],
        [
            "Skip rate (%)",
            fmt_pct(a_skip_rate),
            fmt_pct(b_skip_rate),
            impact_text(float(a_skip_rate), float(b_skip_rate)),
        ],
        [
            "Early accum exits",
            fmt_int(a_early_exits),
            fmt_int(b_early_exits),
            impact_text(float(a_early_exits), float(b_early_exits)),
        ],
        [
            "Micro-batches saved",
            fmt_int(a_mb_saved),
            fmt_int(b_mb_saved),
            impact_text(float(a_mb_saved), float(b_mb_saved)),
        ],
    ]

    table2 = Table(data2, colWidths=[150, 120, 120, 140])
    table2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("GRID", (0, 0), (-1, -1), 1, black),
            ]
        )
    )
    content.append(table2)
    content.append(Spacer(1, 14))

    content.append(PageBreak())

    content.append(Paragraph("4. Visualizations", h1))

    content.append(Paragraph("Figure 1: Total Energy (Wh)", h2))
    content.append(Image(fig_energy, width=300, height=250))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Figure 2: Training Time (s)", h2))
    content.append(Image(fig_time, width=300, height=250))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Figure 3: CO2 Emissions (kg, operational×PUE)", h2))
    content.append(Image(fig_co2, width=300, height=250))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Figure 4: Loss curve", h2))
    if fig_loss and os.path.exists(fig_loss):
        content.append(Image(fig_loss, width=430, height=260))
    else:
        content.append(Paragraph("[No loss history found in trainer_state.json]", body))
    content.append(Spacer(1, 10))

    content.append(Paragraph("5. Notes", h1))
    content.append(
        Paragraph(
            "This report is disaggregated: energy, carbon intensity, and PUE-adjustment are reported separately, "
            "and embodied amortization is optional. GreenTrainer2 efficiency metrics (skipped steps, skip rate, "
            "early accum exits, micro-batches saved) are 0 for baseline runs. "
            "If real-time power measurement is unavailable, the run report may indicate a TDP proxy method.",
            body,
        )
    )

    doc.build(content)
    return pdf_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_a", required=True, help="Path to first run folder")
    p.add_argument("--run_b", required=True, help="Path to second run folder")
    p.add_argument("--label_a", default="Run A")
    p.add_argument("--label_b", default="Run B")
    p.add_argument("--outdir", default="./reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a = load_run(args.run_a, args.label_a)
    b = load_run(args.run_b, args.label_b)

    fig_energy, fig_time, fig_co2, fig_loss = generate_charts(a, b, args.outdir)
    pdf = create_pdf(a, b, fig_energy, fig_time, fig_co2, fig_loss, args.outdir)

    print(
        f"✅ Charts: {fig_energy}, {fig_time}, {fig_co2}"
        + (f", {fig_loss}" if fig_loss else "")
    )
    print(f"✅ PDF: {pdf}")


if __name__ == "__main__":
    main()
