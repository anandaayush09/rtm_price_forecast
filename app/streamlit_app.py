"""Streamlit app: the day-ahead forecast of the RTM price and how it did on recent days.

    uv run streamlit run app/streamlit_app.py

It reads the files the command line writes (``rtm-forecast forecast`` and ``settle``), from the
folders named in the ``production`` section of ``params.yaml`` (set RTM_PARAMS to use another
file). It fits nothing, and no price data ships with the code.
"""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from rtm_forecast import dashboard
from rtm_forecast.config import ProductionParams, load_production_params

LINE = "#2563eb"
PRICE = "#d9480f"


@st.cache_data
def _forecast(folder: str, day: pd.Timestamp, stamp: float) -> pd.DataFrame:
    return dashboard.read_forecast(Path(folder), day)


@st.cache_data
def _history(path: str, stamp: float, cap: float) -> pd.DataFrame:
    calibration = pd.read_parquet(path)
    return dashboard.corrected_history(calibration, cap)


def _cap_axis(cap: float) -> alt.Scale:
    return alt.Scale(domain=[0, cap * 1.05], nice=False)


def forecast_chart(frame: pd.DataFrame, cap: float, actual: pd.DataFrame | None) -> alt.Chart:
    """Point forecast with the 50 and 90 percent intervals, the published price if there is one,
    and the price cap."""
    base = alt.Chart(frame).encode(x=alt.X("start:T", title="Block start (IST)"))
    y = alt.Y("lo90:Q", title="Rs/MWh", scale=_cap_axis(cap))
    wide = base.mark_area(opacity=0.15, color=LINE, interpolate="step-after").encode(
        y=y, y2="hi90:Q"
    )
    narrow = base.mark_area(opacity=0.3, color=LINE, interpolate="step-after").encode(
        y=alt.Y("lo50:Q", scale=_cap_axis(cap)), y2="hi50:Q"
    )
    line = base.mark_line(color=LINE, interpolate="step-after").encode(
        y=alt.Y("point:Q", scale=_cap_axis(cap)),
        tooltip=[
            alt.Tooltip("start:T", title="Block start", format="%H:%M"),
            alt.Tooltip("point:Q", title="Forecast", format=",.0f"),
            alt.Tooltip("lo90:Q", title="90% low", format=",.0f"),
            alt.Tooltip("hi90:Q", title="90% high", format=",.0f"),
        ],
    )
    layers = [wide, narrow, line]
    if actual is not None:
        layers.append(
            alt.Chart(actual)
            .mark_line(color=PRICE, interpolate="step-after")
            .encode(x="start:T", y=alt.Y("actual:Q", scale=_cap_axis(cap)))
        )
    rule = (
        alt.Chart(pd.DataFrame({"cap": [cap]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y=alt.Y("cap:Q", scale=_cap_axis(cap)))
    )
    return alt.layer(*layers, rule).properties(height=380)


def _show_forecast(params: ProductionParams, days: list[pd.Timestamp]) -> None:
    day = st.sidebar.selectbox("Forecast day", days, format_func=lambda d: f"{d:%a %d %b %Y}")
    folder = params.forecast_dir
    stamp = (folder / f"forecast_{day:%Y-%m-%d}.csv").stat().st_mtime
    forecast = _forecast(str(folder), day, stamp)
    actual = None
    if params.calibration_path.exists():
        settled = pd.read_parquet(params.calibration_path)
        settled = settled[(settled["date"] == day) & settled["actual"].notna()]
        if len(settled):
            actual = settled.assign(start=dashboard.block_start(settled["date"], settled["block"]))
    summary = dashboard.day_summary(forecast, params.price_cap)
    first, second, third, fourth = st.columns(4)
    first.metric("Blocks at the cap", f"{summary['blocks_at_cap']:.0f} of {len(forecast)}")
    second.metric("Blocks, cap possible", f"{summary['blocks_possible']:.0f}")
    third.metric("Mean, Rs/MWh", f"{summary['mean_point']:,.0f}")
    fourth.metric(
        "Highest, Rs/MWh",
        f"{summary['highest_point']:,.0f}",
        help=f"Block {summary['highest_block']:.0f}",
    )
    st.altair_chart(forecast_chart(forecast, params.price_cap, actual), width="stretch")
    note = "Line: forecast. Dark band: 50% interval. Light band: 90% interval. Dashed: price cap."
    if actual is not None:
        note += " Orange: the published price."
    st.caption(note)
    shown = forecast.assign(
        Start=forecast["start"].dt.strftime("%H:%M"),
        Expectation=forecast["state"].map(dashboard.STATE_NAMES),
    ).rename(
        columns={
            "block": "Block",
            "point": "Forecast",
            "lo50": "50% low",
            "hi50": "50% high",
            "lo90": "90% low",
            "hi90": "90% high",
        }
    )
    columns = ["Block", "Start", "Forecast", "50% low", "50% high", "90% low", "90% high"]
    with st.expander("All 96 blocks"):
        table = shown[[*columns, "Expectation"]]
        st.dataframe(table.round(0), hide_index=True, width="stretch")
        st.download_button(
            "Download as csv", table.to_csv(index=False), f"forecast_{day:%Y-%m-%d}.csv"
        )


def _show_record(params: ProductionParams) -> None:
    path = params.calibration_path
    if not path.exists():
        st.info("No calibration table yet. Run `rtm-forecast calibrate`, then `settle` each day.")
        return
    history = _history(str(path), path.stat().st_mtime, params.price_cap)
    record = dashboard.daily_record(history)
    if record.empty:
        st.info("Not enough settled days yet for a record.")
        return
    totals = dashboard.record_totals(record)
    first, second, third, fourth = st.columns(4)
    first.metric("Days", f"{totals['days']:.0f}")
    second.metric("Forecast error", f"{totals['model_mae']:,.0f}")
    third.metric(
        "Yesterday's error",
        f"{totals['yesterday_mae']:,.0f}",
        f"{totals['model_mae'] - totals['yesterday_mae']:+,.0f} for the forecast",
        delta_color="inverse",
    )
    fourth.metric("Inside 90% interval", f"{totals['inside90']:.0%}")
    st.caption(
        "Errors are mean absolute errors per block in Rs/MWh, over days that have the day before "
        "in the table. Intervals are corrected the way they are for a new forecast, from the "
        "days before."
    )
    long = record.melt("date", ["model_mae", "yesterday_mae"], "series", "error").assign(
        series=lambda t: t["series"].map({"model_mae": "Forecast", "yesterday_mae": "Yesterday"})
    )
    errors = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Day"),
            y=alt.Y("error:Q", title="Mean absolute error, Rs/MWh"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=["Forecast", "Yesterday"], range=[LINE, "gray"]),
            ),
        )
        .properties(height=260)
    )
    st.altair_chart(errors, width="stretch")
    cover = (
        alt.Chart(record)
        .mark_line(color=LINE)
        .encode(
            x=alt.X("date:T", title="Day"),
            y=alt.Y("inside90:Q", title="Inside the 90% interval", scale=alt.Scale(domain=[0, 1])),
        )
    )
    rule = alt.Chart(pd.DataFrame({"y": [0.9]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((cover + rule).properties(height=200), width="stretch")
    st.dataframe(
        record.sort_values("date", ascending=False).round(
            {"model_mae": 0, "yesterday_mae": 0, "inside50": 3, "inside90": 3}
        ),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="RTM day-ahead forecast", layout="wide")
    params = load_production_params(Path(os.environ.get("RTM_PARAMS", "params.yaml")))
    st.title("RTM price forecast, day ahead")
    st.caption(
        "Forecast of the 96 real-time market blocks of a day, made once the day-ahead market "
        "results of that day are known. Read from local files; no price data ships with the code."
    )
    days = dashboard.forecast_days(params.forecast_dir)
    forecast_tab, record_tab = st.tabs(["Forecast", "Recent days"])
    with forecast_tab:
        if days:
            _show_forecast(params, days)
        else:
            st.info(
                f"No forecast files in `{params.forecast_dir}`. "
                "Fit the model and forecast a day: "
                "`rtm-forecast fit`, `rtm-forecast calibrate`, "
                "`rtm-forecast forecast --day YYYY-MM-DD`"
            )
    with record_tab:
        _show_record(params)


main()
