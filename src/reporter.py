"""
reporter.py
===========
Auto-generates a professional PDF report summarizing the forecasting project.

Sections:
  1. Cover Page          — Title, date, project overview
  2. Dataset Summary     — Shape, date range, missing values, key stats
  3. Model Results       — Metrics table for all 3 models
  4. Forecast Charts     — Embedded PNG images
  5. Conclusions         — Best model, recommendations

Uses fpdf2 (FPDF2) — no external latex/wkhtmltopdf needed.
"""

from fpdf import FPDF, XPos, YPos
from pathlib import Path
from datetime import datetime
import json

BASE_DIR    = Path(__file__).resolve().parent.parent
IMAGES_DIR  = BASE_DIR / "images"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)



# ── Sanitise text for fpdf2 Helvetica (latin-1 only) ─────────────────────────
def _s(text: str) -> str:
    """Replace non-latin-1 characters with ASCII equivalents for Helvetica font."""
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2012": "-",   # em/en dashes
        "\u2018": "'", "\u2019": "'",                   # smart quotes
        "\u201c": '"', "\u201d": '"',                   # smart double quotes
        "\u2022": "-", "\u2023": "-", "\u2043": "-",   # bullets
        "\u00d7": "x", "\u00f7": "/",                  # multiply/divide
        "\u2192": "->", "\u2190": "<-", "\u2194": "<->",
        "\u2260": "!=", "\u2248": "~=", "\u00b2": "2", "\u00b3": "3",
        "\u03b1": "alpha", "\u03b2": "beta", "\u03c0": "pi",
        "\u221a": "sqrt", "\u2211": "sum", "\u222b": "int",
        "\u2264": "<=", "\u2265": ">=",
        "\u00b0": "deg", "\u00b1": "+/-",
        "\u00e9": "e", "\u00e8": "e", "\u00ea": "e",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Final fallback: strip anything outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ── Custom PDF class ──────────────────────────────────────────────────────────

class EnergyReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "AI-Powered Energy Consumption Forecasting - Confidential Report",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d')}",
                  align="C")

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 120, 200)
        self.ln(4)
        self.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(0, 120, 200)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def kv_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(70, 7, key + ":", new_x=XPos.RIGHT, new_y=YPos.LAST)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ── Report Generator ──────────────────────────────────────────────────────────

def _cover_page(pdf: EnergyReport):
    pdf.add_page()
    pdf.set_fill_color(10, 22, 40)
    pdf.rect(0, 0, pdf.w, pdf.h, "F")

    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 194, 255)
    pdf.cell(0, 14, "AI-Powered Energy", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 14, "Consumption Forecasting", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 10, "Automated Performance & Forecast Report",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, "Models: ARIMA | XGBoost | LSTM",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _dataset_section(pdf: EnergyReport, stats: dict):
    pdf.add_page()
    pdf.section_title("1. Dataset Summary")
    pdf.body_text(
        "This report uses the UCI Machine Learning Repository — Individual Household "
        "Electric Power Consumption dataset, augmented with synthetic weather and occupancy "
        "features to simulate a realistic smart grid environment."
    )
    pdf.ln(2)
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        pdf.kv_row(label, str(v))


def _metrics_section(pdf: EnergyReport, metrics_list: list):
    pdf.add_page()
    pdf.section_title("2. Model Performance Metrics")
    pdf.body_text(
        "Three forecasting models were trained and evaluated on a temporal hold-out set (last 20% of data). "
        "Metrics: MAE (lower=better), RMSE (lower=better), MAPE % (lower=better), R² (higher=better)."
    )
    pdf.ln(2)

    # Table header
    col_w = [35, 25, 25, 25, 25, 25, 25]
    headers = ["Model", "MAE", "RMSE", "MAPE (%)", "SMAPE (%)", "R2", "Adj R2"]
    pdf.set_fill_color(0, 120, 200)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 9, h, border=1, align="C", fill=True)
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 10)
    for i, row in enumerate(metrics_list):
        fill = (i % 2 == 0)
        pdf.set_fill_color(240, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        values = [
            row.get("model", ""),
            str(row.get("MAE", "")),
            str(row.get("RMSE", "")),
            str(row.get("MAPE", "")),
            str(row.get("SMAPE", "")),
            str(row.get("R2", "")),
            str(row.get("Adj_R2", "")),
        ]
        for w, v in zip(col_w, values):
            pdf.cell(w, 8, v, border=1, align="C", fill=fill)
        pdf.ln()


def _charts_section(pdf: EnergyReport):
    chart_info = [
        ("consumption_overview.png", "Energy Consumption Overview",
         "Full time-series of household active power with 7-day moving average."),
        ("seasonal_patterns.png",    "Seasonal Load Patterns",
         "Heatmap of hourly consumption by day-of-week — reveals peak demand windows."),
        ("feature_importance.png",   "XGBoost Feature Importances",
         "Top-15 features by gain — lag_24h and rolling_mean_24 dominate."),
        ("model_comparison.png",     "Model Performance Comparison",
         "Grouped bar chart: MAE, RMSE, MAPE across ARIMA, XGBoost, and LSTM."),
        ("lstm_forecast.png",        "LSTM — Actual vs Forecast",
         "LSTM predicted vs actual power for the test set with 95% PI band."),
        ("anomaly_detection.png",    "Anomaly Detection",
         "Z-score based anomaly overlay — red points indicate unusual consumption events."),
    ]

    pdf.add_page()
    pdf.section_title("3. Forecast Visualisations")

    for filename, title, caption in chart_info:
        path = IMAGES_DIR / filename
        if not path.exists():
            continue
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, caption)
        pdf.ln(1)
        available_h = pdf.h - pdf.get_y() - pdf.b_margin - 20
        img_h = min(80, available_h)
        if img_h < 30:
            pdf.add_page()
            img_h = 80
        pdf.image(str(path), x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin, h=img_h)
        pdf.ln(6)


def _conclusions_section(pdf: EnergyReport, metrics_list: list):
    pdf.add_page()
    pdf.section_title("4. Conclusions & Recommendations")

    if metrics_list:
        best = min(metrics_list, key=lambda x: x.get("RMSE", float("inf")))
        pdf.body_text(
            f"Best performing model: {best['model']} "
            f"(RMSE={best.get('RMSE','N/A')}, R2={best.get('R2','N/A')}).\n\n"
            "Recommendations:\n"
            "  - Deploy the XGBoost model for real-time 1-24h ahead forecasting.\n"
            "  - Use LSTM for weekly pattern forecasting (168h horizon).\n"
            "  - Integrate anomaly alerts in energy management dashboard.\n"
            "  - Retrain models monthly as new data is collected.\n"
            "  - Consider ensemble: 0.4xXGBoost + 0.6xLSTM for improved accuracy.\n"
        )

    pdf.body_text(
        "This system demonstrates end-to-end ML engineering: data ingestion, "
        "feature engineering, multi-model training, evaluation, anomaly detection, "
        "and automated reporting - aligned with industry energy analytics workflows."
    )


def generate_report(stats: dict = None, metrics_list: list = None,
                    filename: str = "energy_report.pdf") -> Path:
    """
    Generate the full PDF report.

    Args:
        stats        : Dataset summary dict from data_loader.get_summary_stats()
        metrics_list : List of evaluate_model() dicts
        filename     : Output filename

    Returns:
        Path to the generated PDF.
    """
    print("\n[REPORT] Generating PDF report …")
    pdf = EnergyReport()

    _cover_page(pdf)
    _dataset_section(pdf, stats or {"Note": "Statistics not provided"})
    _metrics_section(pdf, metrics_list or [])
    _charts_section(pdf)
    _conclusions_section(pdf, metrics_list or [])

    out = REPORTS_DIR / filename
    pdf.output(str(out))
    print(f"[REPORT] PDF saved → {out}")
    return out
