"""
data_analysis.py

Practical examples of analyzing structured data using pandas.
Includes:
- Loading CSV files
- Cleaning and enriching data
- Calculating KPIs (revenue, units sold, averages)
- Grouping and aggregating data
- Exporting summary reports
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict

import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent / "data"
SALES_FILE = DATA_DIR / "sales_data.csv"
REPORTS_DIR = pathlib.Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


@dataclass
class SalesSummary:
    total_orders: int
    total_units: int
    total_revenue: float
    avg_order_value: float


def load_sales_data() -> pd.DataFrame:
    """Load the CSV data into a pandas DataFrame."""
    if not SALES_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {SALES_FILE}")

    df = pd.read_csv(SALES_FILE, parse_dates=["date"])
    df["revenue"] = df["units"] * df["unit_price"]
    return df


def summarize_sales(df: pd.DataFrame) -> SalesSummary:
    """Calculate top-line KPIs for the dataset."""
    total_orders = len(df)
    total_units = int(df["units"].sum())
    total_revenue = float(df["revenue"].sum())
    avg_order_value = float(df["revenue"].mean())

    return SalesSummary(
        total_orders=total_orders,
        total_units=total_units,
        total_revenue=total_revenue,
        avg_order_value=avg_order_value,
    )


def revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Return revenue aggregated by sales region."""
    return (
        df.groupby("region")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def top_products(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return the top N products by revenue."""
    return (
        df.groupby("product")[["units", "revenue"]]
        .sum()
        .sort_values(by="revenue", ascending=False)
        .head(n)
        .reset_index()
    )


def salesperson_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Show revenue per salesperson."""
    return (
        df.groupby("salesperson")[["units", "revenue"]]
        .sum()
        .sort_values(by="revenue", ascending=False)
        .reset_index()
    )


def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by month."""
    monthly = (
        df.set_index("date")
        .resample("M")[["units", "revenue"]]
        .sum()
        .reset_index()
    )
    monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
    return monthly[["month", "units", "revenue"]]


def export_report(frames: Dict[str, pd.DataFrame]) -> pathlib.Path:
    """Export a multi-sheet Excel workbook with summary tables."""
    report_path = REPORTS_DIR / "sales_report.xlsx"

    with pd.ExcelWriter(report_path, engine="xlsxwriter") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return report_path


def display_dataframe(title: str, df: pd.DataFrame) -> None:
    """Pretty-print a DataFrame in the terminal."""
    print("\n" + title)
    print("-" * len(title))
    print(df.to_string(index=False))


def run_analysis() -> None:
    """End-to-end analysis workflow."""
    df = load_sales_data()
    summary = summarize_sales(df)

    print("=" * 60)
    print("DATA ANALYSIS: SALES DASHBOARD")
    print("=" * 60)
    print(f"Records Loaded:   {summary.total_orders}")
    print(f"Total Units Sold: {summary.total_units}")
    print(f"Total Revenue:   ${summary.total_revenue:,.2f}")
    print(f"Avg Order Value: ${summary.avg_order_value:,.2f}")

    frames = {
        "Revenue by Region": revenue_by_region(df),
        "Top Products": top_products(df, n=5),
        "Salesperson Performance": salesperson_performance(df),
        "Monthly Trends": monthly_trends(df),
    }

    for title, frame in frames.items():
        display_dataframe(title, frame)

    report_path = export_report(frames)
    print(f"\n📊 Excel report exported to: {report_path.relative_to(pathlib.Path.cwd())}")

    print("\nNext steps:")
    print("  • Connect to a live database or API for real-time analytics")
    print("  • Add data visualizations (matplotlib, seaborn, plotly)")
    print("  • Schedule this script via cron or CI to refresh reports")


if __name__ == "__main__":
    run_analysis()
