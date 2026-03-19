from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


DATA_FILE = Path("flights_clean.csv")
CACHE_DIR = Path(".dash_cache")
CHUNK_SIZE = 250_000

ROUTE_CACHE = CACHE_DIR / "route_agg.pkl"
HOUR_CACHE = CACHE_DIR / "hour_agg.pkl"
MONTH_CACHE = CACHE_DIR / "month_agg.pkl"
MAP_CACHE = CACHE_DIR / "map_agg.pkl"


USECOLS = [
    "Month",
    "Reporting_Airline",
    "OriginState",
    "DestState",
    "Route",
    "ArrDel15",
    "ArrDelay",
    "DepDelay",
    "ScheduledDepHour",
]

DTYPES = {
    "Month": "Int8",
    "Reporting_Airline": "string",
    "OriginState": "string",
    "DestState": "string",
    "Route": "string",
    "ArrDel15": "float32",
    "ArrDelay": "float32",
    "DepDelay": "float32",
    "ScheduledDepHour": "Int8",
}

GROUP_FILTERS = ["Reporting_Airline", "Month", "OriginState", "DestState"]
MONTH_LABELS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": "14px",
    "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
    "padding": "10px",
}


def add_subtitle(fig: go.Figure, subtitle: str) -> go.Figure:
    fig.update_layout(
        title={
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "color": "#111827"},
        }
    )
    fig.add_annotation(
        text=subtitle,
        x=0.02,
        y=1.06,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        font={"size": 12, "color": "#6b7280"},
    )
    return fig


def cache_is_fresh(cache_path: Path) -> bool:
    return cache_path.exists() and cache_path.stat().st_mtime >= DATA_FILE.stat().st_mtime


def finalize_aggregates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [col for col in df.columns if col not in group_cols]
    return (
        df.groupby(group_cols, observed=True, as_index=False)[metric_cols]
        .sum(numeric_only=True)
        .sort_values(group_cols)
    )


def load_or_build_aggregates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(exist_ok=True)

    if all(cache_is_fresh(path) for path in [ROUTE_CACHE, HOUR_CACHE, MONTH_CACHE, MAP_CACHE]):
        return (
            pd.read_pickle(ROUTE_CACHE),
            pd.read_pickle(HOUR_CACHE),
            pd.read_pickle(MONTH_CACHE),
            pd.read_pickle(MAP_CACHE),
        )

    route_parts: list[pd.DataFrame] = []
    hour_parts: list[pd.DataFrame] = []
    month_parts: list[pd.DataFrame] = []
    map_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        DATA_FILE,
        usecols=USECOLS,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk["Reporting_Airline"] = chunk["Reporting_Airline"].fillna("Unknown")
        chunk["OriginState"] = chunk["OriginState"].fillna("Unknown")
        chunk["DestState"] = chunk["DestState"].fillna("Unknown")
        chunk["Route"] = chunk["Route"].fillna("Unknown")
        chunk["ArrDel15"] = chunk["ArrDel15"].fillna(0)
        chunk["ArrDelay"] = chunk["ArrDelay"].fillna(0)
        chunk["DepDelay"] = chunk["DepDelay"].fillna(0)
        chunk["ScheduledDepHour"] = chunk["ScheduledDepHour"].astype("Int8")
        chunk["ScheduledDepHour"] = chunk["ScheduledDepHour"].replace(24, 0)  # midnight fix

        route_parts.append(
            chunk.groupby(GROUP_FILTERS + ["Route"], observed=True, as_index=False).agg(
                flights=("ArrDel15", "size"),
                delayed_flights=("ArrDel15", "sum"),
                arr_delay_total=("ArrDelay", "sum"),
            )
        )
        hour_parts.append(
            chunk.groupby(GROUP_FILTERS + ["ScheduledDepHour"], observed=True, as_index=False).agg(
                flights=("ArrDel15", "size"),
                delayed_flights=("ArrDel15", "sum"),
            )
        )
        month_parts.append(
            chunk.groupby(GROUP_FILTERS, observed=True, as_index=False).agg(
                flights=("ArrDel15", "size"),
                delayed_flights=("ArrDel15", "sum"),
                arr_delay_total=("ArrDelay", "sum"),
            )
        )
        map_parts.append(
            chunk.groupby(GROUP_FILTERS, observed=True, as_index=False).agg(
                flights=("DepDelay", "size"),
                dep_delay_total=("DepDelay", "sum"),
            )
        )

    route_agg = finalize_aggregates(pd.concat(route_parts, ignore_index=True), GROUP_FILTERS + ["Route"])
    hour_agg = finalize_aggregates(
        pd.concat(hour_parts, ignore_index=True), GROUP_FILTERS + ["ScheduledDepHour"]
    )
    month_agg = finalize_aggregates(pd.concat(month_parts, ignore_index=True), GROUP_FILTERS)
    map_agg = finalize_aggregates(pd.concat(map_parts, ignore_index=True), GROUP_FILTERS)

    route_agg.to_pickle(ROUTE_CACHE)
    hour_agg.to_pickle(HOUR_CACHE)
    month_agg.to_pickle(MONTH_CACHE)
    map_agg.to_pickle(MAP_CACHE)

    return route_agg, hour_agg, month_agg, map_agg


def normalize_multi_value(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "ALL")]
    if value == "ALL":
        return []
    return [value]


def apply_filters(
    df: pd.DataFrame,
    airlines,
    months,
    origin_states,
    dest_states,
) -> pd.DataFrame:
    filtered = df

    airline_values = normalize_multi_value(airlines)
    month_values = normalize_multi_value(months)
    origin_values = normalize_multi_value(origin_states)
    dest_values = normalize_multi_value(dest_states)

    if airline_values:
        filtered = filtered.loc[filtered["Reporting_Airline"].isin(airline_values)]
    if month_values:
        filtered = filtered.loc[filtered["Month"].isin(month_values)]
    if origin_values:
        filtered = filtered.loc[filtered["OriginState"].isin(origin_values)]
    if dest_values:
        filtered = filtered.loc[filtered["DestState"].isin(dest_values)]

    return filtered


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#4b5563"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        title={"x": 0.02, "xanchor": "left"},
    )
    return fig


def route_figure(
    route_df: pd.DataFrame,
    airlines,
    months,
    origin_states,
    dest_states,
    min_route_volume: int,
) -> go.Figure:
    filtered = apply_filters(route_df, airlines, months, origin_states, dest_states)
    if filtered.empty:
        return empty_figure("No routes match the selected filters.")

    summary = (
        filtered.groupby("Route", as_index=False)
        .agg(
            flights=("flights", "sum"),
            delayed_flights=("delayed_flights", "sum"),
            arr_delay_total=("arr_delay_total", "sum"),
        )
        .assign(
            delay_rate=lambda d: d["delayed_flights"] / d["flights"],
            avg_arr_delay=lambda d: d["arr_delay_total"] / d["flights"],
        )
    )

    summary = summary.loc[summary["flights"] >= min_route_volume].copy()
    summary = summary.sort_values(["delay_rate", "flights"], ascending=[False, False]).head(15)
    if summary.empty:
        return empty_figure("No routes remain after the minimum route volume filter.")

    summary = summary.sort_values("delay_rate", ascending=True)
    fig = px.bar(
        summary,
        x="delay_rate",
        y="Route",
        orientation="h",
        template="plotly_white",
        text="delay_rate",
        custom_data=["Route", "flights", "avg_arr_delay", "delay_rate"],
        color="delay_rate",
        color_continuous_scale="Tealgrn",
        labels={"delay_rate": "Arrival Delay Rate", "Route": "Origin-Destination Route"},
        title="Routes With the Highest Arrival Delay Risk",
    )
    fig.update_traces(
        texttemplate="%{text:.1%}",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Flights: %{customdata[1]:,.0f}<br>"
            "Average ArrDelay: %{customdata[2]:.1f} min<br>"
            "Delay rate: %{customdata[3]:.1%}<extra></extra>"
        ),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(coloraxis_showscale=False, margin={"l": 80, "r": 20, "t": 76, "b": 50})
    add_subtitle(fig, "Ranks routes by the share of flights arriving 15 or more minutes late.")
    return fig


def hour_figure(hour_df: pd.DataFrame, airlines, months, origin_states, dest_states) -> go.Figure:
    filtered = apply_filters(hour_df, airlines, months, origin_states, dest_states)
    if filtered.empty:
        return empty_figure("No flights match the selected filters.")

    summary = (
        filtered.groupby("ScheduledDepHour", as_index=False)
        .agg(flights=("flights", "sum"), delayed_flights=("delayed_flights", "sum"))
        .assign(
            delay_rate=lambda d: d["delayed_flights"] / d["flights"],
            hour_label=lambda d: d["ScheduledDepHour"].apply(
                lambda h: "12 AM" if h == 0 else "12 PM" if h == 12 else f"{h} AM" if h < 12 else f"{h - 12} PM"
            ),
        )
        .sort_values("ScheduledDepHour")
    )

    fig = px.line(
        summary,
        x="ScheduledDepHour",
        y="delay_rate",
        markers=True,
        template="plotly_white",
        labels={"ScheduledDepHour": "Scheduled Departure Time", "delay_rate": "Arrival Delay Rate"},
        title="Arrival Delay Risk Across the Day",
    )
    fig.update_xaxes(
        tickvals=summary["ScheduledDepHour"],
        ticktext=summary["hour_label"],
    )
    fig.update_traces(line={"width": 3, "color": "#264653"}, marker={"size": 8})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin={"l": 60, "r": 20, "t": 76, "b": 50})
    add_subtitle(fig, "Shows how often flights arrive late depending on scheduled departure time.")
    return fig


def month_figure(month_df: pd.DataFrame, airlines, months, origin_states, dest_states) -> go.Figure:
    filtered = apply_filters(month_df, airlines, months, origin_states, dest_states)
    if filtered.empty:
        return empty_figure("No flights match the selected filters.")

    summary = (
        filtered.groupby("Month", as_index=False)
        .agg(
            flights=("flights", "sum"),
            delayed_flights=("delayed_flights", "sum"),
            arr_delay_total=("arr_delay_total", "sum"),
        )
        .assign(
            delay_rate=lambda d: d["delayed_flights"] / d["flights"],
            avg_arr_delay=lambda d: d["arr_delay_total"] / d["flights"],
            month_label=lambda d: d["Month"].map(MONTH_LABELS),
        )
        .sort_values("Month")
    )

    fig = px.line(
        summary,
        x="month_label",
        y="delay_rate",
        markers=True,
        template="plotly_white",
        labels={"month_label": "Month of 2024", "delay_rate": "Arrival Delay Rate"},
        title="Monthly Pattern in Arrival Delays",
        custom_data=["avg_arr_delay", "flights"],
    )
    fig.update_traces(
        line={"width": 3, "color": "#2a9d8f"},
        marker={"size": 8},
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Delay rate: %{y:.1%}<br>"
            "Average ArrDelay: %{customdata[0]:.1f} min<br>"
            "Flights: %{customdata[1]:,.0f}<extra></extra>"
        ),
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin={"l": 60, "r": 20, "t": 76, "b": 50})
    add_subtitle(fig, "Highlights seasonality and system stress over the 2024 travel calendar.")
    return fig


def map_figure(map_df: pd.DataFrame, airlines, months, origin_states, dest_states) -> go.Figure:
    filtered = apply_filters(map_df, airlines, months, origin_states, dest_states)
    if filtered.empty:
        return empty_figure("No flights match the selected filters.")

    summary = (
        filtered.groupby("OriginState", as_index=False)
        .agg(
            flights=("flights", "sum"),
            dep_delay_total=("dep_delay_total", "sum"),
        )
        .assign(avg_dep_delay=lambda d: d["dep_delay_total"] / d["flights"])
    )

    fig = px.choropleth(
        summary,
        locations="OriginState",
        locationmode="USA-states",
        color="avg_dep_delay",
        scope="usa",
        template="plotly_white",
        color_continuous_scale="YlOrRd",
        title="Geographic Delay Map by Origin State",
        labels={"avg_dep_delay": "Average Departure Delay (min)", "OriginState": "Origin State"},
        custom_data=["flights", "avg_dep_delay"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Flight volume: %{customdata[0]:,.0f}<br>"
            "Average departure delay: %{customdata[1]:.1f} min<extra></extra>"
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 76, "b": 20},
        coloraxis_colorbar={"title": "Avg Dep Delay", "ticksuffix": " min"},
        geo={"showlakes": False, "showland": True, "landcolor": "#f8fafc", "bgcolor": "white"},
    )
    add_subtitle(
        fig,
        "States are colored by average departure delay to show where disruption is most concentrated.",
    )
    return fig


route_agg_df, hour_agg_df, month_agg_df, map_agg_df = load_or_build_aggregates()

airline_options = sorted(route_agg_df["Reporting_Airline"].dropna().unique().tolist())
month_options = sorted(route_agg_df["Month"].dropna().unique().tolist())
origin_state_options = sorted(route_agg_df["OriginState"].dropna().unique().tolist())
dest_state_options = sorted(route_agg_df["DestState"].dropna().unique().tolist())

app = Dash(__name__)
app.title = "Traveler Delay Risk Explorer"

app.layout = html.Div(
    style={
        "maxWidth": "1500px",
        "margin": "0 auto",
        "padding": "28px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "backgroundColor": "#f3f4f6",
    },
    children=[
        html.Div(
            [
                html.H1(
                    "Traveler Delay Risk Explorer",
                    style={"marginBottom": "8px", "color": "#111827", "fontSize": "34px"},
                ),
                html.P(
                    "An interactive view of how route, departure time, seasonality, and operational causes relate to arrival delay risk in U.S. domestic flights during 2024.",
                    style={"marginTop": 0, "marginBottom": "4px", "color": "#4b5563", "fontSize": "16px"},
                ),
                html.P(
                    "Use the filters to focus on a traveler segment, then compare risk patterns across routes, time of day, and delay causes.",
                    style={"marginTop": 0, "color": "#6b7280", "fontSize": "14px"},
                ),
            ],
            style={
                "backgroundColor": "white",
                "padding": "22px 24px",
                "borderRadius": "16px",
                "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
                "marginBottom": "18px",
            },
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, minmax(0, 1fr))",
                "gap": "14px",
                "backgroundColor": "white",
                "padding": "18px",
                "borderRadius": "16px",
                "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
                "marginBottom": "18px",
            },
            children=[
                html.Div(
                    [
                        html.Label("Airline"),
                        dcc.Dropdown(
                            id="airline-filter",
                            options=[{"label": airline, "value": airline} for airline in airline_options],
                            multi=True,
                            placeholder="All airlines",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Month"),
                        dcc.Dropdown(
                            id="month-filter",
                            options=[{"label": MONTH_LABELS[m], "value": m} for m in month_options],
                            multi=True,
                            placeholder="All months",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Origin State"),
                        dcc.Dropdown(
                            id="origin-filter",
                            options=[{"label": state, "value": state} for state in origin_state_options],
                            multi=True,
                            placeholder="All origin states",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Destination State"),
                        dcc.Dropdown(
                            id="dest-filter",
                            options=[{"label": state, "value": state} for state in dest_state_options],
                            multi=True,
                            placeholder="All destination states",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Minimum Route Volume"),
                        dcc.Slider(
                            id="route-volume-filter",
                            min=100,
                            max=5000,
                            step=100,
                            value=500,
                            marks={100: "100", 500: "500", 1000: "1k", 2500: "2.5k", 5000: "5k"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style={"padding": "0 8px"},
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.2fr 1fr", "gap": "18px", "marginBottom": "18px"},
            children=[
                html.Div(
                    dcc.Graph(id="route-chart", config={"displayModeBar": False}),
                    style=CARD_STYLE,
                ),
                html.Div(
                    dcc.Graph(id="hour-chart", config={"displayModeBar": False}),
                    style=CARD_STYLE,
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"},
            children=[
                html.Div(
                    dcc.Graph(id="month-chart", config={"displayModeBar": False}),
                    style=CARD_STYLE,
                ),
                html.Div(
                    dcc.Graph(id="map-chart", config={"displayModeBar": False}),
                    style=CARD_STYLE,
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("route-chart", "figure"),
    Output("hour-chart", "figure"),
    Output("month-chart", "figure"),
    Output("map-chart", "figure"),
    Input("airline-filter", "value"),
    Input("month-filter", "value"),
    Input("origin-filter", "value"),
    Input("dest-filter", "value"),
    Input("route-volume-filter", "value"),
)
def update_dashboard(airlines, months, origin_states, dest_states, min_route_volume):
    route_fig = route_figure(
        route_agg_df, airlines, months, origin_states, dest_states, min_route_volume
    )
    hour_fig = hour_figure(hour_agg_df, airlines, months, origin_states, dest_states)
    month_fig = month_figure(month_agg_df, airlines, months, origin_states, dest_states)
    map_fig = map_figure(map_agg_df, airlines, months, origin_states, dest_states)
    return route_fig, hour_fig, month_fig, map_fig


if __name__ == "__main__":
    app.run(debug=True)