"""
visual_styles.py - Institutional Light & Dark Design System
CSS themes for Streamlit, a validated categorical palette and a Plotly theme helper.
"""

INSTITUTIONAL_LIGHT_CSS = """
<style>
/* App canvas in light mode */
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

div[data-testid="stMetricLabel"] p {
    color: #64748B !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}

div[data-testid="stMetricValue"] {
    color: #0284C7 !important;
    font-size: 1.75rem !important;
    font-weight: 800;
}

/* Tab Navigation */
.stTabs [data-baseweb="tab-list"] {
    background-color: #F1F5F9;
    padding: 5px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    color: #64748B;
    font-weight: 600;
    border-radius: 6px;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background-color: #0284C7 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0;
}

/* Insight Cards */
.card-insight {
    background: #FFFFFF;
    border-left: 4px solid #0284C7;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    padding: 14px 18px;
    border-radius: 8px;
    margin-top: 10px;
    margin-bottom: 15px;
    font-size: 0.88rem;
    color: #334155;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}

.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}

.badge-real {
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #A7F3D0;
}

.badge-synth {
    background: #EEF2FF;
    color: #4F46E5;
    border: 1px solid #C7D2FE;
}
</style>
"""

INSTITUTIONAL_DARK_CSS = """
<style>
.stApp {
    background-color: #0B0F19;
    color: #F1F5F9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

div[data-testid="stMetricLabel"] p {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
    font-size: 1.75rem !important;
    font-weight: 800;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(15, 23, 42, 0.7);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #0284C7 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 10px rgba(2, 132, 199, 0.35);
}

section[data-testid="stSidebar"] {
    background-color: #070B13 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

.card-insight {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #38BDF8;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 14px 18px;
    border-radius: 8px;
    margin-top: 10px;
    margin-bottom: 15px;
    font-size: 0.88rem;
    color: #CBD5E1;
}

.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}

.badge-real {
    background: rgba(52, 211, 153, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.4);
}

.badge-synth {
    background: rgba(129, 140, 248, 0.15);
    color: #818CF8;
    border: 1px solid rgba(129, 140, 248, 0.4);
}
</style>
"""

# Categorical palette (validated for colour-vision deficiency, adjacent pairs, both modes).
# Slots are assigned in fixed order and follow the entity, never its rank.
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
COMPONENT_LABELS = {"cgst": "CGST", "sgst": "SGST", "igst": "IGST", "cess": "Cess"}


def palette(is_dark: bool = False) -> dict:
    cat = CATEGORICAL_DARK if is_dark else CATEGORICAL_LIGHT
    return {
        "categorical": cat,
        "primary": cat[0],
        "accent": cat[1],
        "components": {"cgst": cat[0], "sgst": cat[1], "igst": cat[2], "cess": cat[3]},
        "muted": "#8a8984" if is_dark else "#9a9993",
        "text": "#f0efec" if is_dark else "#0b0b0b",
        "text_secondary": "#c3c2b7" if is_dark else "#52514e",
        "sequential": "Blues",
    }


def apply_plotly_theme(fig, title: str = None, height: int = 450, is_dark: bool = False):
    """Applies the light/dark chart theme. Leaves an existing title alone unless one is passed."""
    if is_dark:
        bg_paper, bg_plot = "rgba(0,0,0,0)", "rgba(15, 23, 42, 0.45)"
        text_color, subtext_color = "#E2E8F0", "#94A3B8"
        grid_color, legend_bg = "rgba(255, 255, 255, 0.08)", "rgba(15, 23, 42, 0.75)"
    else:
        bg_paper, bg_plot = "#FFFFFF", "#FCFCFB"
        text_color, subtext_color = "#0F172A", "#64748B"
        grid_color, legend_bg = "rgba(0, 0, 0, 0.07)", "rgba(255, 255, 255, 0.85)"

    fig.update_layout(
        height=height,
        paper_bgcolor=bg_paper,
        plot_bgcolor=bg_plot,
        colorway=CATEGORICAL_DARK if is_dark else CATEGORICAL_LIGHT,
        font={"family": "-apple-system, BlinkMacSystemFont, Roboto, sans-serif", "color": text_color},
        title_font={"size": 15, "color": text_color},
        margin={"l": 50, "r": 30, "t": 60, "b": 45},
        legend={"font": {"color": text_color}, "bgcolor": legend_bg},
        hoverlabel={"font_size": 12},
        bargap=0.15,
    )
    fig.update_xaxes(gridcolor=grid_color, linecolor=grid_color, tickfont={"color": subtext_color}, zeroline=False,
                     automargin=True)
    fig.update_yaxes(gridcolor=grid_color, linecolor=grid_color, tickfont={"color": subtext_color}, zeroline=False,
                     automargin=True)
    if title is not None:
        fig.update_layout(title={"text": f"<b>{title}</b>"})
    elif fig.layout.title.text:
        fig.update_layout(title={"text": f"<b>{fig.layout.title.text}</b>"})
    return fig
