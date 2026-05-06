import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html

# ── Load data ──────────────────────────────────────────────────────────────────
# CSV must sit in the same directory as this script
CSV_PATH = os.path.join(os.path.dirname(__file__), "hotel_bookings_cleaned.csv")
df = pd.read_csv(CSV_PATH)

# ── Design tokens ──────────────────────────────────────────────────────────────
BLUE        = "#4C78A8"
RED         = "#E45756"
GREEN       = "#54A24B"
BG          = "#F0F2F5"
CARD_BG     = "#FFFFFF"
TEXT_DARK   = "#1E2D3D"
TEXT_SUB    = "#64748B"
GRID_COLOR  = "#E8ECF0"
FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"


# ══════════════════════════════════════════════════════════════════════════════
# 1. CANCELLATION RATE BY LEAD TIME  — area / line chart
# ══════════════════════════════════════════════════════════════════════════════
def build_lead_time_chart():
    df_lt = df[df["LeadTime"] <= 365].copy()
    df_lt["LeadTimeBin"] = (df_lt["LeadTime"] // 15) * 15

    agg = (
        df_lt.groupby("LeadTimeBin")["IsCanceled"]
        .agg(cancel_rate="mean", count="count")
        .reset_index()
    )
    agg = agg[agg["count"] >= 50].copy()
    agg["cancel_pct"] = agg["cancel_rate"] * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agg["LeadTimeBin"],
        y=agg["cancel_pct"],
        fill="tozeroy",
        fillcolor="rgba(76, 120, 168, 0.15)",
        line=dict(color=BLUE, width=3),
        mode="lines+markers",
        marker=dict(size=6, color=BLUE, line=dict(color=CARD_BG, width=1.5)),
        name="Cancellation Rate",
        hovertemplate="<b>Lead time:</b> %{x} days<br><b>Cancel rate:</b> %{y:.1f}%<extra></extra>",
    ))

    overall = df["IsCanceled"].mean() * 100
    fig.add_hline(
        y=overall,
        line=dict(color=RED, width=1.5, dash="dash"),
        annotation_text=f"Overall avg {overall:.1f}%",
        annotation_position="top right",
        annotation_font=dict(color=RED, size=12),
    )

    fig.update_layout(
        title=dict(
            text="<b>Cancellation Rate by Lead Time</b>",
            font=dict(size=18, color=TEXT_DARK, family=FONT_FAMILY),
            x=0.01,
        ),
        xaxis=dict(
            title="Lead Time (days, binned every 15 days)",
            gridcolor=GRID_COLOR,
            showgrid=True,
            tickfont=dict(size=12),
            title_font=dict(size=13),
        ),
        yaxis=dict(
            title="Cancellation Rate (%)",
            gridcolor=GRID_COLOR,
            showgrid=True,
            range=[0, 90],
            ticksuffix="%",
            tickfont=dict(size=12),
            title_font=dict(size=13),
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=70, r=40, t=70, b=65),
        hovermode="x unified",
        font=dict(family=FONT_FAMILY),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASS IMBALANCE — donut chart
# ══════════════════════════════════════════════════════════════════════════════
def build_class_imbalance_chart():
    counts  = df["IsCanceled"].value_counts().sort_index()
    not_can = counts.get(0, 0)
    can     = counts.get(1, 0)
    total   = not_can + can

    fig = go.Figure(go.Pie(
        labels=["Not Cancelled", "Cancelled"],
        values=[not_can, can],
        hole=0.60,
        marker=dict(
            colors=[GREEN, RED],
            line=dict(color=CARD_BG, width=4),
        ),
        textinfo="percent",
        textfont=dict(size=15, color=CARD_BG),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        pull=[0, 0.05],
        direction="clockwise",
        sort=False,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Booking Outcome Distribution (Class Imbalance)</b>",
            font=dict(size=18, color=TEXT_DARK, family=FONT_FAMILY),
            x=0.01,
        ),
        annotations=[
            dict(
                text=f"<b>{total:,}</b>",
                x=0.5, y=0.55,
                font=dict(size=22, color=TEXT_DARK, family=FONT_FAMILY),
                showarrow=False,
            ),
            dict(
                text="bookings",
                x=0.5, y=0.43,
                font=dict(size=14, color=TEXT_SUB, family=FONT_FAMILY),
                showarrow=False,
            ),
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=13, family=FONT_FAMILY),
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=20, r=20, t=70, b=50),
        font=dict(family=FONT_FAMILY),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. SPECIAL REQUESTS VS CANCELLATION RATE — grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def build_special_requests_chart():
    agg = (
        df.groupby("TotalOfSpecialRequests")["IsCanceled"]
        .agg(cancel_rate="mean", count="count")
        .reset_index()
    )
    agg["cancel_pct"]     = agg["cancel_rate"] * 100
    agg["not_cancel_pct"] = 100 - agg["cancel_pct"]
    agg["label"]          = agg["TotalOfSpecialRequests"].astype(str) + " request(s)"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Not Cancelled",
        x=agg["label"],
        y=agg["not_cancel_pct"],
        marker_color=GREEN,
        marker_line=dict(color=CARD_BG, width=1),
        text=agg["not_cancel_pct"].round(1).astype(str) + "%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=13, color=CARD_BG),
        hovertemplate="<b>%{x}</b><br>Not cancelled: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Cancelled",
        x=agg["label"],
        y=agg["cancel_pct"],
        marker_color=RED,
        marker_line=dict(color=CARD_BG, width=1),
        text=agg["cancel_pct"].round(1).astype(str) + "%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=13, color=CARD_BG),
        hovertemplate="<b>%{x}</b><br>Cancelled: %{y:.1f}%<extra></extra>",
    ))

    for _, row in agg.iterrows():
        fig.add_annotation(
            x=row["label"],
            y=104,
            text=f"n={int(row['count']):,}",
            showarrow=False,
            font=dict(size=11, color=TEXT_SUB, family=FONT_FAMILY),
            yref="y",
        )

    fig.update_layout(
        title=dict(
            text="<b>Special Requests vs Cancellation Rate</b>",
            font=dict(size=18, color=TEXT_DARK, family=FONT_FAMILY),
            x=0.01,
        ),
        barmode="group",
        xaxis=dict(
            title="Number of Special Requests",
            gridcolor=GRID_COLOR,
            tickfont=dict(size=12),
            title_font=dict(size=13),
        ),
        yaxis=dict(
            title="Percentage of Bookings (%)",
            gridcolor=GRID_COLOR,
            range=[0, 115],
            ticksuffix="%",
            tickfont=dict(size=12),
            title_font=dict(size=13),
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=13, family=FONT_FAMILY),
        ),
        margin=dict(l=70, r=30, t=80, b=65),
        bargap=0.22,
        bargroupgap=0.06,
        font=dict(family=FONT_FAMILY),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. MONTHLY CANCELLATION RATE vs AVERAGE ADR — dual-axis line chart
# ══════════════════════════════════════════════════════════════════════════════
def build_monthly_dual_axis_chart():
    MONTH_ORDER = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    agg = (
        df.groupby("ArrivalDateMonth")
        .agg(cancel_rate=("IsCanceled", "mean"), avg_adr=("ADR", "mean"))
        .reset_index()
    )
    agg["ArrivalDateMonth"] = pd.Categorical(
        agg["ArrivalDateMonth"], categories=MONTH_ORDER, ordered=True
    )
    agg = agg.sort_values("ArrivalDateMonth").reset_index(drop=True)
    agg["cancel_pct"]  = agg["cancel_rate"] * 100
    agg["month_short"] = agg["ArrivalDateMonth"].astype(str).str[:3]

    # Tight padded axis ranges so both lines show clear variation
    cr_min, cr_max = agg["cancel_pct"].min(), agg["cancel_pct"].max()
    cr_pad = (cr_max - cr_min) * 0.35
    adr_min, adr_max = agg["avg_adr"].min(), agg["avg_adr"].max()
    adr_pad = (adr_max - adr_min) * 0.35

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=agg["month_short"],
            y=agg["cancel_pct"],
            name="Cancellation Rate (%)",
            mode="lines+markers",
            line=dict(color=RED, width=3),
            marker=dict(size=10, color=RED, line=dict(color=CARD_BG, width=2)),
            hovertemplate="<b>%{x}</b><br>Cancel rate: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=agg["month_short"],
            y=agg["avg_adr"],
            name="Average ADR (€)",
            mode="lines+markers",
            line=dict(color=BLUE, width=3, dash="dot"),
            marker=dict(size=10, color=BLUE, line=dict(color=CARD_BG, width=2)),
            hovertemplate="<b>%{x}</b><br>Avg ADR: €%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text="<b>Monthly Cancellation Rate vs Average ADR</b>",
            font=dict(size=18, color=TEXT_DARK, family=FONT_FAMILY),
            x=0.01,
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            font=dict(size=13, family=FONT_FAMILY),
        ),
        margin=dict(l=70, r=80, t=80, b=65),
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=12)),
        font=dict(family=FONT_FAMILY),
    )
    fig.update_yaxes(
        title_text="Cancellation Rate (%)",
        secondary_y=False,
        gridcolor=GRID_COLOR,
        range=[cr_min - cr_pad, cr_max + cr_pad],
        ticksuffix="%",
        tickfont=dict(size=12),
        title_font=dict(size=13),
    )
    fig.update_yaxes(
        title_text="Average ADR (€)",
        secondary_y=True,
        showgrid=False,
        range=[adr_min - adr_pad, adr_max + adr_pad],
        tickprefix="€",
        tickfont=dict(size=12),
        title_font=dict(size=13),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Dash app layout
# ══════════════════════════════════════════════════════════════════════════════
app = Dash(__name__)
app.title = "Hotel Bookings Dashboard"

CARD = {
    "backgroundColor": CARD_BG,
    "borderRadius": "12px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
    "padding": "8px 16px 16px 16px",
    "marginBottom": "24px",
}

ROW_2 = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "gap": "24px",
    "marginBottom": "24px",
}

app.layout = html.Div(
    style={
        "backgroundColor": BG,
        "minHeight": "100vh",
        "padding": "36px 48px",
        "fontFamily": FONT_FAMILY,
    },
    children=[
        # Header
        html.Div([
            html.H1(
                "Hotel Bookings Dashboard",
                style={
                    "color": TEXT_DARK,
                    "margin": "0 0 6px 0",
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "letterSpacing": "-0.3px",
                },
            ),
            html.P(
                f"Dataset: {len(df):,} bookings  •  "
                f"Overall cancellation rate: {df['IsCanceled'].mean()*100:.1f}%  •  "
                f"Years: {df['ArrivalDateYear'].min()}–{df['ArrivalDateYear'].max()}",
                style={"color": TEXT_SUB, "margin": 0, "fontSize": "14px"},
            ),
        ], style={"marginBottom": "28px"}),

        # Chart 1 — Lead Time (full width)
        html.Div([
            dcc.Graph(
                figure=build_lead_time_chart(),
                config={"displayModeBar": False},
                style={"height": "420px"},
            ),
        ], style=CARD),

        # Charts 2 & 3 — Donut + Grouped bar (50/50)
        html.Div([
            html.Div([
                dcc.Graph(
                    figure=build_class_imbalance_chart(),
                    config={"displayModeBar": False},
                    style={"height": "460px"},
                ),
            ], style=CARD),
            html.Div([
                dcc.Graph(
                    figure=build_special_requests_chart(),
                    config={"displayModeBar": False},
                    style={"height": "460px"},
                ),
            ], style=CARD),
        ], style=ROW_2),

        # Chart 4 — Monthly dual-axis (full width)
        html.Div([
            dcc.Graph(
                figure=build_monthly_dual_axis_chart(),
                config={"displayModeBar": False},
                style={"height": "440px"},
            ),
        ], style=CARD),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
