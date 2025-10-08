# streamlit_app.py
# Bally products on Cettire – Comparison & Analytics UI
# Data source: plain CSV `match_final.csv` with columns:
# c_link,domain,category,c_title,c_retail_price,c_sale_price,c_image-src,c_season_tag,c_product_url,c_product_id,
# matchlink,m_title,m_retail_price,m_sale_price,m_image-src,m_product_id,m_season_tag
#
# Definitions:
# - Final price (both sides): final = sale if present else retail; if both missing => NA (excluded from numeric stats).
# - Price difference (Diff Cettire – Match): positive => Cettire more expensive; negative => Cettire cheaper.
# - Pct_Cheaper_Cettire: % rows (with both finals & match not OOS) where Cettire final < Match final.
# - Pct_Cettire_Discount: % rows where c_sale < c_retail (on Cettire).
# - Pct_Match_Discount: % rows where m_sale < m_retail (on matched site), excluding OOS.

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
# ... other imports ...


def _bally_altair_theme():
    return {
        "config": {
            "background": "#ffffff",
            "view": {"fill": "#ffffff", "stroke": "#e5e5e5"},
            "axis": {
                "labelColor": "#111111",
                "titleColor": "#111111",
                "gridColor": "#f1f3f5",
                "domainColor": "#d0d4da",
            },
            "legend": {"labelColor": "#111111", "titleColor": "#111111"},
            "title": {"color": "#111111"},
        }
    }


alt.themes.register("bally_light", _bally_altair_theme)
alt.themes.enable("bally_light")

st.set_page_config(page_title="Bally – Cettire Benchmark", layout="wide")

# Global styling inspired by bally.com.au palette
GLOBAL_CSS = """
<style>
html, body {
    background: #ffffff !important;
}

:root {
    --bally-page: #ffffff;
    --bally-shell: #ffffff;
    --bally-ink: #111111;
    --bally-border: #e5e5e5;
    --background-color: #ffffff !important;
    --secondary-background-color: #f6f6f6 !important;
    --text-color: #111111 !important;
    --primary-color: #111111 !important;
    --primary-color-rgb: 17,17,17 !important;
}

html, body, .stApp, #root, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bally-page) !important;
    color: var(--bally-ink);
    font-family: "Helvetica Neue", "Arial", sans-serif;
    color-scheme: only light;
}

[data-testid="stHeader"] {
    background: var(--bally-page) !important;
    color: var(--bally-ink) !important;
}

header[data-testid="stHeader"] {
    background: var(--bally-page);
    border-bottom: 1px solid var(--bally-border);
    box-shadow: none;
}

.main .block-container {
    padding-top: 2rem;
    background: var(--bally-page);
}

div[data-testid="stSidebar"] {
    background: var(--bally-shell);
    border-right: 1px solid var(--bally-border);
}

section[data-testid="stSidebar"] {
    background: var(--bally-shell) !important;
}

div[data-testid="stSidebarCollapsedControl"] {
    background: var(--bally-shell) !important;
}

div[data-testid="stSidebarNav"] {
    background: var(--bally-shell) !important;
}

section[data-testid="stSidebar"] > div:first-child {
    background: var(--bally-shell) !important;
}

div[data-testid="stSidebar"] *, div[data-testid="stSidebar"] label {
    color: var(--bally-ink) !important;
}

div[data-testid="stSidebar"] button {
    background: #111111;
    border: none;
    color: #ffffff !important;
    font-size: 0.95rem;
    border-radius: 999px;
    padding: 0.35rem 1.2rem;
    text-decoration: none;
}

div[data-testid="stSidebar"] button:hover {
    background: #242424;
}

/* Harmonise selectboxes / multiselects with clean borders */
div[data-baseweb="select"] > div {
    background: var(--bally-page);
    border-radius: 12px;
    border: 1px solid #d0d4da;
}

div[data-baseweb="select"] input {
    color: var(--bally-ink) !important;
}

[data-baseweb="menu"] {
    background: #ffffff !important;
    color: var(--bally-ink) !important;
    border: 1px solid #d0d4da !important;
    box-shadow: 0 12px 32px rgba(0,0,0,0.12) !important;
}

[data-baseweb="option"] {
    background: #ffffff !important;
    color: var(--bally-ink) !important;
}

[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"] {
    background: #f2f3f5 !important;
    color: var(--bally-ink) !important;
}

div[data-testid="stSidebar"] .stCheckbox, div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] input {
    color: var(--bally-ink) !important;
}

div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span,
div[data-testid="stSidebar"] .stMarkdown {
    color: var(--bally-ink) !important;
}

div[data-testid="stSidebar"] .stRadio > div > label {
    color: var(--bally-ink) !important;
}

div[data-baseweb="radio"] {
    color: var(--bally-ink) !important;
}

div[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--bally-page);
    border-radius: 12px;
    border: 1px solid #d0d4da;
    color: var(--bally-ink) !important;
}

.stDownloadButton button {
    border-radius: 999px;
    background: #111111;
    color: #ffffff;
    border: none;
    padding: 0.35rem 1.5rem;
}

div[data-testid="stMetricValue"] {
    color: var(--bally-ink);
}

div[data-testid="stDataFrame"] {
    background: var(--bally-page);
    border-radius: 16px;
    border: 1px solid var(--bally-border);
}

div[data-testid="stDataFrame"] table {
    background: var(--bally-page) !important;
    color: var(--bally-ink) !important;
}

div[data-testid="stDataFrame"] th {
    background: #f2f2f2 !important;
    color: var(--bally-ink) !important;
}

div[data-testid="stDataFrame"] td {
    color: var(--bally-ink) !important;
}

[data-testid="stDataFrame"] [data-testid="stToolbar"] button {
    background: #ffffff !important;
    color: var(--bally-ink) !important;
    border: 1px solid #d0d4da !important;
}

.stDataFrame div[data-testid="StyledDataFrame"] {
    background: var(--bally-page) !important;
    color: var(--bally-ink) !important;
}

.stDataFrame div[role="columnheader"] {
    background: #f2f3f5 !important;
    color: var(--bally-ink) !important;
    border-bottom: 1px solid var(--bally-border);
}

.stDataFrame div[role="row"] {
    background: var(--bally-page) !important;
    color: var(--bally-ink) !important;
}

.stDataFrame div[role="gridcell"] {
    border-color: rgba(0,0,0,0.05) !important;
}

.stDataFrame [class*="row_heading"],
.stDataFrame [class*="col_heading"],
.stDataFrame [class*="blank"] {
    background: #f2f3f5 !important;
    color: var(--bally-ink) !important;
}

.stDataFrame div[role="grid"],
.stDataFrame div[role="grid"] table,
.stDataFrame div[role="rowgroup"],
[data-testid="stDataFrame"] div[data-testid="stStyledFullWidthTable"],
[data-testid="stDataFrame"] div[data-testid="stStyledFullWidthTable"] table {
    background: var(--bally-page) !important;
    color: var(--bally-ink) !important;
}

[data-testid="stDataFrame"] .ag-root-wrapper,
[data-testid="stDataFrame"] .ag-root-wrapper-body,
[data-testid="stDataFrame"] .ag-center-cols-container,
[data-testid="stDataFrame"] .ag-header,
[data-testid="stDataFrame"] .ag-row,
[data-testid="stDataFrame"] .ag-cell {
    background-color: var(--bally-page) !important;
    color: var(--bally-ink) !important;
    border-color: rgba(0,0,0,0.05) !important;
}

[data-testid="stDataFrame"] .ag-header,
[data-testid="stDataFrame"] .ag-header-row,
[data-testid="stDataFrame"] .ag-header-cell {
    background-color: #f2f3f5 !important;
    color: var(--bally-ink) !important;
    border-color: rgba(0,0,0,0.06) !important;
}

.stDataFrame tbody tr {
    background: var(--bally-page) !important;
}

.vega-embed, .vega-embed canvas, .vega-embed svg {
    background: var(--bally-page) !important;
}

.vg-tooltip {
    color: var(--bally-ink);
}

.pagination-bar {
    position: sticky;
    top: 76px;
    z-index: 6;
    background: var(--bally-page);
    padding: 0.5rem 0 0.75rem;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}

.pagination-bar .stNumberInput input {
    border-radius: 999px;
    border: 1px solid #d0d4da;
}

.page-status {
    padding-top: 1.8rem;
    font-size: 14px;
    font-weight: 600;
}

.page-meta {
    padding-top: 1.8rem;
    font-size: 13px;
    color: #6b7280;
    text-align: right;
}
</style>
"""

DEFAULT_FILTER_STATE = {
    "flt_cats": ["All"],
    "flt_sites": ["All"],
    "flt_seasons": ["All"],
    "flt_sort": "Default (Bally-first)",
    "flt_naoos": True,
    "flt_bucket": "All",
    "flt_min": "",
    "flt_max": "",
}


def reset_filter_state() -> None:
    """Restore sidebar controls to their default selections."""
    for key, value in DEFAULT_FILTER_STATE.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value

# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)
# --- Domain/category ranking for default sort (Bally-first) ---

def strip_www(d: str) -> str:
    s = str(d or "")
    return s[4:] if s.startswith("www.") else s



def weighted_discount_by_domain_category(df: pd.DataFrame) -> pd.DataFrame:
    # keep only rows with both prices and not OOS
    mr = df["m_retail_price"].apply(to_num)
    ms = df["m_sale_price"].apply(to_num)
    mask_valid = (~df["oos_match"]) & mr.notna() & ms.notna()

    t = df[mask_valid].copy()
    t["retail"] = mr[mask_valid]
    t["sale"]   = ms[mask_valid]
    t["disc_amt"]   = (t["retail"] - t["sale"]).clip(lower=0)
    t["disc_flag"]  = t["disc_amt"] > 0

    # sum only discounted retail and amounts (price-weighted)
    t["retail_disc_only"] = np.where(t["disc_flag"], t["retail"], 0.0)
    t["disc_amt_only"]    = np.where(t["disc_flag"], t["disc_amt"], 0.0)

    g = (
        t.groupby(["domain", "category"], dropna=False)
         .agg(
             Rows=("disc_flag", "size"),
             Discounted=("disc_flag", "sum"),
             SumRetail=("retail_disc_only", "sum"),
             SumDisc=("disc_amt_only", "sum"),
         )
         .reset_index()
    )

    g["PctWeighted"] = np.where(g["SumRetail"] > 0, 100.0 * g["SumDisc"] / g["SumRetail"], np.nan)
    # keep only cells that actually have discounted SKUs
    g = g[g["Discounted"] > 0].copy()

    g["domain_clean"] = g["domain"].str.replace(r"^www\.", "", regex=True)
    return g

def _prep_match_discount_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Prep rows for match-side discount analytics."""
    t = df.copy()
    t["mr"] = t["m_retail_price"].apply(to_num)
    t["ms"] = t["m_sale_price"].apply(to_num)
    t["oos"] = t["m_season_tag"].apply(is_oos)
    # discounted if not OOS, both prices exist, and sale < retail
    t["disc"] = (~t["oos"]) & t["mr"].notna() & t["ms"].notna() & (t["ms"] < t["mr"])
    # price-weighted discount % for discounted rows only
    t["disc_pct"] = np.where(t["disc"], (t["mr"] - t["ms"]) / t["mr"] * 100.0, np.nan)
    t["site"] = t["domain"].map(strip_www)
    t["category"] = t["category"].fillna("Unknown")
    return t[["domain", "site", "category", "mr", "ms", "disc", "disc_pct"]]

def cat_domain_share_discount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Share of SKUs discounted (%), by (domain, category).
    Counts all rows; those not discounted (including missing price data or OOS) count as 0.
    """
    t = _prep_match_discount_rows(df)
    g = t.groupby(["domain", "site", "category"], dropna=False)["disc"]
    share = g.mean().mul(100.0).reset_index().rename(columns={"disc": "PctDiscounted"})
    # also keep the discounted count for labels/tooltip
    n_disc = g.sum().reset_index().rename(columns={"disc": "DiscCount"})
    out = share.merge(n_disc, on=["domain", "site", "category"], how="left")
    return out

def cat_domain_price_weighted_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price-weighted average discount (%) among discounted SKUs only:
      sum(mr - ms) / sum(mr) * 100  within each (domain, category).
    """
    t = _prep_match_discount_rows(df)
    d = t[t["disc"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["domain", "site", "category", "AvgDiscount", "DiscCount"])
    grp = d.groupby(["domain", "site", "category"], dropna=False)
    avg = (grp.apply(lambda g: ((g["mr"] - g["ms"]).sum() / g["mr"].sum()) * 100.0)
              .reset_index(name="AvgDiscount"))
    # discounted SKU count for context
    avg["DiscCount"] = grp.size().values
    return avg


def domain_group_rank(domain: str) -> int:
    """
    0 = www.bally.com.au
    1 = other bally.* domains
    2 = other retailers (e.g., davidjones, etc.)
    3 = farfetch (always last)
    """
    d = (domain or "").lower()
    if "farfetch.com" in d:
        return 3
    if "www.bally.com.au" in d:
        return 0
    # any other Bally site
    if "bally." in d or "ballyofswitzerland" in d:
        return 1
    return 2  # other partners

def category_rank_for_bally_au(category: str) -> int:
    """
    Inside Bally AU only: Bags -> 0, Shoes -> 1, others -> 2.
    """
    c = (category or "").strip().lower()
    if c == "bags":
        return 0
    if c == "shoes":
        return 1
    return 2

def s(x) -> str:
    """Safe string for HTML/text."""
    if x is None or is_nan(x):
        return ""
    return str(x)

def to_num(x) -> float | None:
    """Robust numeric caster. Returns None if not parseable."""
    if x is None or is_nan(x):
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    try:
        cleaned = (
            str(x)
            .replace("AUD", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .replace("AED", "")
            .replace(",", "")
            .strip()
        )
        if cleaned == "" or cleaned.lower() == "na":
            return None
        return float(cleaned)
    except Exception:
        return None

def money(x: float | None) -> str:
    if x is None:
        return "—"
    return f"AUD$ {x:,.2f}"

def pick_final(sale, retail) -> float | None:
    s_val = to_num(sale)
    r_val = to_num(retail)
    if s_val is not None:
        return s_val
    if r_val is not None:
        return r_val
    return None

def discount_percent(sale, retail) -> float | None:
    s_val = to_num(sale)
    r_val = to_num(retail)
    if s_val is None or r_val is None:
        return None
    if s_val >= r_val:
        return None
    return max(0.0, (r_val - s_val) / r_val * 100.0)

def is_oos(tag: str | None) -> bool:
    return s(tag).strip().lower() in {"out of stock", "oos"}

SITE_ORDER = {
    "www.bally.com.au": 0,
    "www.bally.ch": 1,
    "www.bally.ae": 2,
    "www.bally.eu": 3,
    "www.davidjones.com": 4,
    "www.farfetch.com": 5,
    "ballyofswitzerland.tw": 6,
}

def site_priority(domain) -> int:
    d = s(domain).lower()
    for k, v in SITE_ORDER.items():
        if k in d:
            return v
    return 999

def diff_pill_html(diff: float | None) -> str:
    if diff is None or (isinstance(diff, float) and pd.isna(diff)):
        return ""
    color = "red" if diff < 0 else "green" if diff > 0 else ""
    sign  = "-" if diff < 0 else "+" if diff > 0 else ""
    return f'<span class="price-pill {color}">{sign}AUD$ {abs(diff):,.2f}</span>'

# --------------------------------------------------------------------------------------
# Load & enrich
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "NaN", "nan"]
    )

    df.columns = [c.strip() for c in df.columns]

    # finals
    df["c_final"] = [pick_final(r.get("c_sale_price"), r.get("c_retail_price")) for _, r in df.iterrows()]
    df["m_final"] = [
        None if is_oos(r.get("m_season_tag")) else pick_final(r.get("m_sale_price"), r.get("m_retail_price"))
        for _, r in df.iterrows()
    ]
    df["oos_match"] = df["m_season_tag"].apply(is_oos)

    # price diff: Cettire – Match
    def diff_row(row):
        cf, mf = row["c_final"], row["m_final"]
        if cf is None or mf is None:
            return None
        return cf - mf
    df["diff_c_minus_m"] = df.apply(diff_row, axis=1)

    # season union (for filtering)
    def season_union(row):
        a, b = s(row.get("c_season_tag")).strip(), s(row.get("m_season_tag")).strip()
        if a and b and a.lower() != b.lower():
            return f"{a} / {b}"
        return a or b or "No tag"
    df["season_union"] = df.apply(season_union, axis=1)

    # site priority
    df["site_priority"] = df["domain"].apply(site_priority)

    # fill text fields (avoid NaN in HTML)
    for c in [
        "c_title", "c_image-src", "c_product_url", "c_link", "c_season_tag",
        "m_title", "m_image-src", "matchlink", "m_season_tag", "category", "domain"
    ]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    return df

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
CARD_CSS = """
<style>
.app-wrap { max-width: 1280px; margin: 0 auto; padding: 0 2rem 3rem; }

.count-pill { font-size:12px; color:#374151; padding:4px 12px; border-radius:999px; background:#f1f3f5; display:inline-block; }

.comp-row { display:flex; gap:32px; margin-bottom:24px; }
.comp-tile { flex: 1; background:var(--bally-page); border:1px solid var(--bally-border); border-radius:18px; box-shadow:0 12px 24px rgba(17,17,17,0.04); overflow:hidden; transition: box-shadow 0.2s ease; }
.comp-tile:hover { box-shadow:0 18px 32px rgba(17,17,17,0.07); }
.comp-tile a { color: inherit; text-decoration: none; display:block; }
.comp-body { padding:18px 22px 20px; }

.img-wrap { height: 280px; display:flex; align-items:center; justify-content:center; background:var(--bally-shell); }
.img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; }

.site-label { font-size: 12px; color:#6b7280; margin-bottom:4px; letter-spacing:0.08em; text-transform:uppercase; }
.title { font-weight:600; font-size:17px; line-height:1.25; margin:6px 0 12px 0; color:var(--bally-ink); }
.title a { text-decoration: underline; }

.badge { display:inline-block; font-size:12px; padding:2px 10px; border-radius:999px; background:#f7f2eb; color:#5b2c0c; border:1px solid #eadac6; }

.price-line { display:flex; align-items:baseline; gap:12px; }
.strike { color:#94a3b8; text-decoration: line-through; }
.price-accent { color:#111827; font-weight:700; font-size:17px; }
.pct-pill { font-size:12px; border-radius:8px; padding:2px 6px; background:#fef3f2; color:#b42318; border:1px solid #fecdca; }

.diff-note { font-size:12px; color:#475569; margin:12px 4px 0; }

.oos { font-weight:700; color:#b42318; }
.dim { color:#94a3b8; }

/* price diff pill colours */
.price-pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;line-height:18px}
.price-pill.red   {background:#FEE2E2;border:1px solid #FCA5A5;color:#B42318;}   /* Cettire cheaper */
.price-pill.green {background:#E6F4EA;border:1px solid #B7E1C0;color:#1B5E20;}   /* Cettire higher  */
</style>
"""

# --------------------------------------------------------------------------------------
# Product tile HTML (single side)
# --------------------------------------------------------------------------------------
def tile_html(
    site_label: str,
    url: str | None,
    title: str | None,
    img: str | None,
    retail: float | None,
    sale: float | None,
    season: str | None,
    oos: bool = False
) -> str:
    t = s(title)
    link = s(url) or "#"
    image = s(img)
    badge = f'<span class="badge" title="{s(season)}">{s(season)}</span>' if s(season) else ""

    final = pick_final(sale, retail)
    strike_html = ""
    price_html = ""
    pct_html = ""

    if oos:
        price_html = '<span class="oos">Out of stock</span>'
    else:
        if final is not None:
            d = discount_percent(sale, retail)
            if d is not None:
                # sale < retail
                strike_html = f'<span class="strike">{money(to_num(retail))}</span>'
                price_html = f'<span class="price-accent">{money(final)}</span>'
                pct_html = f'<span class="pct-pill">-{d:.0f}%</span>'
            else:
                price_html = f'<span class="price-accent">{money(final)}</span>'
        else:
            price_html = '<span class="dim">—</span>'

    body = f"""
    <div class="panel-body comp-body">
      <div class="site-label">{s(site_label)}</div>
      <div class="title">{s(t)}</div>
      {badge}
      <div class="price-line">{strike_html}{price_html}{pct_html}</div>
    </div>
    """

    img_block = f"""
    <div class="img-wrap">
        <img src="{image}" alt="{s(t)}" loading="lazy" />
    </div>
    """

    # Entire tile is a link
    html = f"""
    <div class="comp-tile">
      <a href="{link}" target="_blank" rel="noopener">
        {img_block}
        {body}
      </a>
    </div>
    """
    return html

# --------------------------------------------------------------------------------------
# One comparison row (left Cettire, right Matched)
# --------------------------------------------------------------------------------------
def comp_card_html(r: pd.Series) -> str:
    # left (Cettire)
    c_title = r.get("c_title")
    c_img = r.get("c_image-src")
    c_retail = to_num(r.get("c_retail_price"))
    c_sale = to_num(r.get("c_sale_price"))
    c_url = r.get("c_product_url") or r.get("c_link")  # prefer product_url, fallback to c_link
    c_season = r.get("c_season_tag")

    # right (match)
    m_title = r.get("m_title")
    m_img = r.get("m_image-src")
    m_retail = to_num(r.get("m_retail_price"))
    m_sale = to_num(r.get("m_sale_price"))
    m_url = r.get("matchlink")
    m_season = r.get("m_season_tag")
    m_oos = bool(r.get("oos_match"))

    left = tile_html("Cettire", c_url, c_title, c_img, c_retail, c_sale, c_season, oos=False)
    right = tile_html(s(r.get("domain") or "Match"), m_url, m_title, m_img, m_retail, m_sale, m_season, oos=m_oos)

    # difference note (red/green pill)
    diff = r.get("diff_c_minus_m")
    diff_line = ""
    if diff is not None and (not m_oos) and r.get("m_final") is not None and r.get("c_final") is not None:
        note = f'Diff (Cettire – Match): {diff_pill_html(float(diff))}'
        diff_line = f'<div class="diff-note">{note}</div>'

    row = f"""
    <div class="comp-row">
      {left}
      {right}
    </div>
    {diff_line}
    """
    return row

# --------------------------------------------------------------------------------------
# Filters UI (comparison tab)
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Filters UI (comparison tab)
# --------------------------------------------------------------------------------------

def comparison_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    st.subheader("Products", divider=False)
    st.caption("Use the filters on the left. Pagination shows **50 comparisons per page**.")

    # ---- safe accessor ----
    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(dtype=object)

    # ---- session defaults so “All” is selected on first load ----
    for key, default in DEFAULT_FILTER_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default.copy() if isinstance(default, list) else default

    # ---- helper: expand “All” ----
    def expand_all(selected: list[str], all_vals: list[str]) -> list[str]:
        return all_vals if (not selected) or ("All" in selected) else selected

    with st.sidebar:
        title_col, clear_col = st.columns([2, 1])
        with title_col:
            st.markdown("### Filters")
        with clear_col:
            if st.button("Clear all filters", key="btn_clear_filters"):
                reset_filter_state()
                rerun = getattr(st, "rerun", None)
                if callable(rerun):
                    rerun()
                else:
                    exp_rerun = getattr(st, "experimental_rerun", None)
                    if callable(exp_rerun):
                        exp_rerun()

        cats    = sorted([c for c in col("category").dropna().unique().tolist() if c])
        sites   = sorted([d for d in col("domain").dropna().unique().tolist() if d])
        seasons = sorted([t for t in col("season_union").dropna().unique().tolist() if t])

        # Multiselects with an explicit “All” option
        sel_cats_raw = st.multiselect("Category", ["All", *cats], key="flt_cats")
        sel_sites_raw = st.multiselect("Matched site", ["All", *sites], key="flt_sites")
        sel_seasons_raw = st.multiselect("Season tag (either side)", ["All", *seasons], key="flt_seasons")

        sel_cats    = expand_all(sel_cats_raw, cats)
        sel_sites   = expand_all(sel_sites_raw, sites)
        sel_seasons = expand_all(sel_seasons_raw, seasons)

        st.markdown("### Sorting")
        sort_options = [
            "Default (Bally-first)",
            "Max price diff",
            "Cettire final (low → high)",
            "Cettire final (high → low)",
            "Match final (low → high)",
            "Match final (high → low)",
        ]
        sort_choice = st.selectbox("Sort products by", sort_options, key="flt_sort")

        include_na_oos = st.checkbox("Include NA / OOS", key="flt_naoos")

        st.markdown("### Price filter (Cettire final)")
        bucket = st.selectbox(
            "Price bucket",
            ["All", "$0–$250", "$250–$500", "$500–$1000", "$1000–$2000", "$2000+"],
            key="flt_bucket"
        )
        cmin, cmax = st.columns(2)
        with cmin:
            manual_min = st.text_input("Manual min (AUD)", key="flt_min")
        with cmax:
            manual_max = st.text_input("Manual max (AUD)", key="flt_max")

    # ---- required columns check (fast bail-out with friendly msg) ----
    required = {"domain","category","season_union","c_final","m_final","oos_match","diff_c_minus_m","site_priority"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning("Your file is missing required columns: " + ", ".join(missing))
        st.info("Please upload a full dataset or adjust your filters.")
        return df.iloc[0:0], {}

    # ---- Apply filters ----
    dff = df.copy()
    if sel_cats:    dff = dff[dff["category"].isin(sel_cats)]
    if sel_sites:   dff = dff[dff["domain"].isin(sel_sites)]
    if sel_seasons: dff = dff[dff["season_union"].isin(sel_seasons)]

    if not include_na_oos:
        dff = dff[(dff["c_final"].notna()) & (dff["m_final"].notna()) & (~dff["oos_match"])]

    # Price bucket + manual min/max
    def in_bucket(v, bkt):
        if v is None: return False
        if bkt == "All": return True
        if bkt == "$0–$250": return v < 250
        if bkt == "$250–$500": return 250 <= v < 500
        if bkt == "$500–$1000": return 500 <= v < 1000
        if bkt == "$1000–$2000": return 1000 <= v < 2000
        if bkt == "$2000+": return v >= 2000
        return True

    dff = dff[dff["c_final"].apply(lambda v: in_bucket(v, bucket))]

    lo = to_num(manual_min) if manual_min else None
    hi = to_num(manual_max) if manual_max else None
    if lo is not None: dff = dff[dff["c_final"].apply(lambda x: (x is not None) and (x >= lo))]
    if hi is not None: dff = dff[dff["c_final"].apply(lambda x: (x is not None) and (x <= hi))]

    # ---- Empty / missing-domain guard BEFORE sorting ----
    if dff.empty or ("domain" not in dff.columns) or dff["domain"].isna().all():
        st.info("Ciao, no results with the current filters — please select a broader filter :')")
        return dff.iloc[0:0], {
            "include_na_oos": include_na_oos, "bucket": bucket,
            "manual_min": manual_min, "manual_max": manual_max, "sort": sort_choice,
            "sel_cats": sel_cats_raw, "sel_sites": sel_sites_raw, "sel_seasons": sel_seasons_raw,
        }

    # ---- Sort (Bally-first default) ----
    if sort_choice == "Default (Bally-first)":
        tmp = dff.copy()
        tmp["site_rank"] = tmp["domain"].apply(domain_group_rank)
        tmp["cat_rank"] = tmp.apply(
            lambda r: category_rank_for_bally_au(r["category"]) if (r["domain"] == "www.bally.com.au") else 99,
            axis=1
        )
        tmp["gap"] = tmp["diff_c_minus_m"].abs()
        dff = tmp.sort_values(
            by=["site_rank", "cat_rank", "gap"],
            ascending=[True, True, False],
            na_position="last"
        )
    elif sort_choice == "Max price diff":
        dff = dff.sort_values(["diff_c_minus_m", "site_priority"], ascending=[False, True], na_position="last")
    elif sort_choice == "Cettire final (low → high)":
        dff = dff.sort_values(["c_final", "site_priority"], ascending=[True, True], na_position="last")
    elif sort_choice == "Cettire final (high → low)":
        dff = dff.sort_values(["c_final", "site_priority"], ascending=[False, True], na_position="last")
    elif sort_choice == "Match final (low → high)":
        dff = dff.sort_values(["m_final", "site_priority"], ascending=[True, True], na_position="last")
    elif sort_choice == "Match final (high → low)":
        dff = dff.sort_values(["m_final", "site_priority"], ascending=[False, True], na_position="last")

    controls = {
        "include_na_oos": include_na_oos,
        "bucket": bucket,
        "manual_min": manual_min,
        "manual_max": manual_max,
        "sort": sort_choice,
        "sel_cats": sel_cats_raw,
        "sel_sites": sel_sites_raw,
        "sel_seasons": sel_seasons_raw,
    }
    return dff, controls


# --------------------------------------------------------------------------------------
# Analytics builders
# --------------------------------------------------------------------------------------
def numeric_mask(df: pd.DataFrame) -> pd.Series:
    return (df["c_final"].notna()) & (df["m_final"].notna()) & (~df["oos_match"])
def discounting_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    By-domain discounting on both sides + average % discount on the MATCH site.
    - Pct_Cettire_Discount: share of Cettire rows where c_sale < c_retail
    - Pct_Match_Discount  : share of MATCH rows where m_sale < m_retail (OOS counted as not discounted)
    - AvgDisc_Match       : average (m_retail - m_sale)/m_retail*100 over discounted rows (ignores NaN)
    """
    t = df.copy()

    # --- Cettire side: discounted? (sale < retail)
    t["c_disc"] = t.apply(
        lambda r: (to_num(r.get("c_sale_price")) is not None
                   and to_num(r.get("c_retail_price")) is not None
                   and to_num(r.get("c_sale_price")) < to_num(r.get("c_retail_price"))),
        axis=1
    )

    # --- Match side: discounted? (sale < retail), treat OOS as not discounted
    def match_is_discount(row) -> bool:
        if is_oos(row.get("m_season_tag")):
            return False
        ms, mr = to_num(row.get("m_sale_price")), to_num(row.get("m_retail_price"))
        return (ms is not None) and (mr is not None) and (ms < mr)

    t["m_disc"] = t.apply(match_is_discount, axis=1)

    # --- Match side: discount % for discounted rows; NaN otherwise (so mean ignores them)
    def match_disc_pct(row):
        if not row["m_disc"]:
            return np.nan
        ms, mr = to_num(row.get("m_sale_price")), to_num(row.get("m_retail_price"))
        if ms is None or mr is None or mr <= 0:
            return np.nan
        return (mr - ms) / mr * 100.0

    t["m_disc_pct"] = t.apply(match_disc_pct, axis=1)

    grp = t.groupby("domain", dropna=False)
    res = pd.DataFrame({
        "Rows": grp.size(),
        "Pct_Cettire_Discount": grp["c_disc"].mean() * 100.0,   # share of SKUs on sale (Cettire)
        "Pct_Match_Discount":   grp["m_disc"].mean() * 100.0,   # share of SKUs on sale (Match)
        "AvgDisc_Match":        grp["m_disc_pct"].mean()        # average % discount (Match), discounted rows only
    }).reset_index()

    return res.sort_values("Rows", ascending=False)

def tidy_for_charts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for numeric charts: both finals present and match not OOS."""
    mask = numeric_mask(df)
    t = df[mask].copy()
    if t.empty:
        return t
    t["Domain"] = t["domain"]
    t["Category"] = t["category"]
    t["Diff"] = t["diff_c_minus_m"]
    t["CettireFinal"] = t["c_final"]
    t["MatchFinal"] = t["m_final"]
    t["CettireURL"] = t["c_product_url"].where(t["c_product_url"] != "", t["c_link"])
    t["MatchURL"] = t["matchlink"]
    t["CTitle"] = t["c_title"]
    t["MTitle"] = t["m_title"]
    return t

def site_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df[numeric_mask(df)].copy()
    if work.empty:
        return pd.DataFrame(columns=["domain", "Rows", "Average_Difference", "Median_Difference", "Pct_Cheaper_Cettire"])
    grp = work.groupby("domain", dropna=False)
    res = pd.DataFrame({
        "Rows": grp.size(),
        "Average_Difference": grp["diff_c_minus_m"].mean(),
        "Median_Difference": grp["diff_c_minus_m"].median(),
        "Pct_Cheaper_Cettire": grp.apply(lambda g: np.mean(g["c_final"] < g["m_final"]) * 100.0)
    }).reset_index()
    return res.sort_values("Rows", ascending=False)

def bally_au_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df[numeric_mask(df) & (df["domain"].str.contains("www.bally.com.au", na=False))].copy()
    if work.empty:
        return pd.DataFrame(columns=["category", "Rows", "Average_Difference", "Median_Difference", "Pct_Cheaper_Cettire"])
    grp = work.groupby("category", dropna=False)
    res = pd.DataFrame({
        "Rows": grp.size(),
        "Average_Difference": grp["diff_c_minus_m"].mean(),
        "Median_Difference": grp["diff_c_minus_m"].median(),
        "Pct_Cheaper_Cettire": grp.apply(lambda g: np.mean(g["c_final"] < g["m_final"]) * 100.0)
    }).reset_index()
    return res.sort_values("Rows", ascending=False)

def top_discrepancies(df: pd.DataFrame, n=15) -> pd.DataFrame:
    work = df[numeric_mask(df) & (df["domain"].str.contains("www.bally.com.au", na=False))].copy()
    if work.empty:
        return pd.DataFrame()
    work["Cettire URL"] = work["c_product_url"].where(work["c_product_url"] != "", work["c_link"])
    cols = [
        "category", "domain", "c_title", "m_title",
        "c_final", "m_final", "diff_c_minus_m", "Cettire URL", "matchlink"
    ]
    out = (work
           .sort_values("diff_c_minus_m", ascending=False)
           .head(n)[cols]
           .rename(columns={
                "c_title":"Cettire title",
                "m_title":"Match title",
                "c_final":"Cettire final (AUD)",
                "m_final":"Match final (AUD)",
                "diff_c_minus_m":"Diff (Cettire – Match)",
                "matchlink":"Match URL"
            }))
    return out



# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_analytics(df: pd.DataFrame):
    st.header("Bally SKUs on Cettire.com")
    st.caption("Comprehensive Products analysis by AU E-Commerce Analytics team")
    st.caption("Data scraped: 1–4 Oct 2025 (AUD). Internal use only.")

    # --- headline metrics ---
    total_rows = len(df)
    oos_count = int(df["oos_match"].sum())

    # Matched rows = rows that have a valid match URL (regardless of prices/OOS)
    match_url_present = (
        df.get("matchlink", pd.Series([""] * len(df))).astype(str)
        .str.startswith(("http://", "https://"))
    )
    matched_rows = int(match_url_present.sum())

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Total rows", f"{total_rows:,}")
    with colB:
        st.metric("Cettire SKUs", f"{total_rows:,}")
    with colC:
        st.metric("Matched rows", f"{matched_rows:,}")
    with colD:
        st.metric("OOS (match)", f"{oos_count:,}")

    # --- site-wise table ---
    st.subheader("Site-wise comparison")
    st.write(
        "**Average Difference** = mean(Cettire final – Match final) using rows with both prices and non-OOS. "
        "Positive means Cettire is more expensive; negative means cheaper."
    )
    ss = site_summary(df)
    st.dataframe(ss, use_container_width=True)
    st.download_button(
        "Download site summary (CSV)",
        ss.to_csv(index=False).encode("utf-8"),
        "site_summary.csv",
        "text/csv",
    )

    # --- charts: distribution / scatter / top abs diff ---
    t = tidy_for_charts(df)
    if not t.empty:
        st.markdown("### Distribution of price differences")
        st.caption("Histogram of (Cettire final – Match final). Vertical line at 0 means parity.")
        hist = (
            alt.Chart(t)
            .mark_bar()
            .encode(
                x=alt.X("Diff:Q", bin=alt.Bin(maxbins=40), title="Difference (Cettire – Match) in AUD"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("count()", title="Rows")],
            )
            .properties(height=220)
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#b42318").encode(x="x:Q")
        st.altair_chart(hist + zero, use_container_width=True)

        st.markdown("### Price relationship (Cettire vs Match)")
        st.caption("Each point is a SKU; dashed line y=x means equal price.")
        maxv = float(pd.Series([t["CettireFinal"].max(), t["MatchFinal"].max()]).max())
        line_df = pd.DataFrame({"x": [0, maxv], "y": [0, maxv]})
        pts = (
            alt.Chart(t)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("MatchFinal:Q", title="Match final (AUD)"),
                y=alt.Y("CettireFinal:Q", title="Cettire final (AUD)"),
                color=alt.Color("Domain:N", title="Site"),
                tooltip=[
                    alt.Tooltip("Domain:N", title="Site"),
                    alt.Tooltip("CTitle:N", title="Cettire title"),
                    alt.Tooltip("MTitle:N", title="Match title"),
                    alt.Tooltip("CettireFinal:Q", title="Cettire (AUD)", format=",.2f"),
                    alt.Tooltip("MatchFinal:Q", title="Match (AUD)", format=",.2f"),
                    alt.Tooltip("Diff:Q", title="Diff", format=",.2f"),
                ],
            )
            .properties(height=320)
        )
        xy = alt.Chart(line_df).mark_line(color="#475569", strokeDash=[6, 6]).encode(x="x:Q", y="y:Q")
        st.altair_chart(pts + xy, use_container_width=True)

        st.markdown("### Top 20 absolute price differences (bar view)")
        top20 = t.assign(abs_diff=t["Diff"].abs()).sort_values("abs_diff", ascending=False).head(20)
        top20["Title"] = top20["CTitle"].where(top20["CTitle"] != "", top20["MTitle"])
        top20["TitleShort"] = top20["Title"].str.slice(0, 48)
        bar = (
            alt.Chart(top20)
            .mark_bar()
            .encode(
                x=alt.X("abs_diff:Q", title="Absolute difference (AUD)"),
                y=alt.Y("TitleShort:N", sort="-x", title=None),
                color=alt.Color("Domain:N", legend=None),
                tooltip=[
                    alt.Tooltip("Domain:N", title="Site"),
                    alt.Tooltip("Title:N"),
                    alt.Tooltip("Diff:Q", title="Diff (Cettire–Match)", format=",.2f"),
                    alt.Tooltip("CettireFinal:Q", title="Cettire", format=",.2f"),
                    alt.Tooltip("MatchFinal:Q", title="Match", format=",.2f"),
                ],
            )
            .properties(height=520)
        )
        st.altair_chart(bar, use_container_width=True)

    # --- Bally AU category summary ---
    st.subheader("Bally AU – category difference (select site)")
    st.write("Breakdown for **www.bally.com.au** only (rows where both finals present and match is not OOS).")
    ca = bally_au_category_summary(df)
    st.dataframe(ca, use_container_width=True)
    st.download_button(
        "Download bally.com.au category summary (CSV)",
        ca.to_csv(index=False).encode("utf-8"),
        "bally_au_category_summary.csv",
        "text/csv",
    )

    # --- Top discrepancies (Bally AU) ---
    st.subheader("Top price discrepancies (Bally AU match)")
    st.write("Rows with largest **(Cettire – Match)**. Links open in new tabs.")
    topd = top_discrepancies(df, n=15)
    if not topd.empty:
        try:
            st.dataframe(
                topd,
                use_container_width=True,
                column_config={
                    "Cettire URL": st.column_config.LinkColumn("Cettire URL"),
                    "Match URL": st.column_config.LinkColumn("Match URL"),
                },
            )
        except Exception:
            st.dataframe(topd, use_container_width=True)
    st.download_button(
        "Download discrepancies (CSV)",
        topd.to_csv(index=False).encode("utf-8"),
        "top_discrepancies.csv",
        "text/csv",
    )

    # --- Discounting: table + charts ---
    st.subheader("Discounting summary (by site)")
    st.write(
        "**Cettire discount** = % where `sale < retail` on Cettire; "
        "**Match discount** = % where `sale < retail` on matched site (excludes OOS)."
    )
    disc = discounting_summary(df)
    st.dataframe(disc, use_container_width=True)
    st.download_button(
        "Download discounting (CSV)",
        disc.to_csv(index=False).encode("utf-8"),
        "discounting_summary.csv",
        "text/csv",
    )

    if not disc.empty:
        disc_plot = disc.copy()
        disc_plot["site"] = disc_plot["domain"].str.replace(r"^www\.", "", regex=True)

        # (A) Share discounted (% of SKUs) – Match side
        share_df = disc_plot.sort_values("Pct_Match_Discount", ascending=False).copy()
        share_df["DiscCount"] = (share_df["Pct_Match_Discount"] * share_df["Rows"] / 100.0).round().astype(int)

        st.markdown("### Discounting by site (share of SKUs on sale)")
        bar_disc = (
            alt.Chart(share_df)
            .mark_bar(size=36)
            .encode(
                x=alt.X("site:N", axis=alt.Axis(title=None, labelAngle=0)),
                y=alt.Y("Pct_Match_Discount:Q", title="% of SKUs discounted", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("site:N", title="Site"),
                    alt.Tooltip("Rows:Q", title="Total SKUs"),
                    alt.Tooltip("DiscCount:Q", title="Discounted SKUs"),
                    alt.Tooltip("Pct_Match_Discount:Q", title="% SKUs discounted", format=".1f"),
                    alt.Tooltip("Pct_Cettire_Discount:Q", title="% SKUs discounted (Cettire)", format=".1f"),
                ],
            )
            .properties(height=300)
        )
        labels_disc = bar_disc.mark_text(baseline="bottom", dy=-4, fontSize=12).encode(
            text=alt.Text("Pct_Match_Discount:Q", format=".1f")
        )
        st.altair_chart(bar_disc + labels_disc, use_container_width=True)

        # (B) Average discount among discounted SKUs – Match side
        st.markdown("### Average discount on matched sites (only discounted SKUs)")
        avg_df = disc_plot.sort_values("AvgDisc_Match", ascending=False).copy()
        avg_df["DiscCount"] = (avg_df["Pct_Match_Discount"] * avg_df["Rows"] / 100.0).round().astype(int)
        avg_df["label"] = avg_df["AvgDisc_Match"].map(lambda v: f"{v:.1f}%") + " • n=" + avg_df["DiscCount"].astype(str)
        maxy = max(10.0, float(avg_df["AvgDisc_Match"].max() or 0)) * 1.15

        bar_avg = (
            alt.Chart(avg_df)
            .mark_bar(size=36)
            .encode(
                x=alt.X("site:N", axis=alt.Axis(title=None, labelAngle=0)),
                y=alt.Y("AvgDisc_Match:Q", title="Average discount (%)", scale=alt.Scale(domain=[0, maxy])),
                tooltip=[
                    alt.Tooltip("site:N", title="Site"),
                    alt.Tooltip("AvgDisc_Match:Q", title="Avg discount (%)", format=".1f"),
                    alt.Tooltip("DiscCount:Q", title="Discounted SKUs"),
                    alt.Tooltip("Rows:Q", title="Total SKUs"),
                    alt.Tooltip("Pct_Match_Discount:Q", title="% SKUs discounted", format=".1f"),
                ],
            )
            .properties(height=320)
        )
        labels_avg = alt.Chart(avg_df).mark_text(baseline="bottom", dy=-4, fontSize=12).encode(
            x="site:N", y="AvgDisc_Match:Q", text="label:N"
        )
        st.altair_chart(bar_avg + labels_avg, use_container_width=True)

        st.markdown("### Category × domain (price-weighted average discount %)")

    # Avg discount (%) among discounted SKUs only; includes DiscCount
    avg_mat = cat_domain_price_weighted_avg(df)   # ['domain','site','category','AvgDiscount','DiscCount']

    # We only need domain/site/category from the share table to keep keys aligned.
    share_mat = cat_domain_share_discount(df)[["domain", "site", "category"]]

    # Merge without duplicate DiscCount columns
    heat_df = avg_mat.merge(share_mat, on=["domain", "site", "category"], how="left")
    heat_df["DiscCount"] = heat_df["DiscCount"].fillna(0).astype(int)

    if not heat_df.empty:
        # Category order by discounted volume (more active first)
        cat_order = (
            heat_df.groupby("category")["DiscCount"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        # Show label only when > 0
        heat_df["label"] = heat_df["AvgDiscount"].apply(
            lambda v: f"{v:.1f}%" if pd.notna(v) and v > 0 else ""
        )

        heat = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("category:N", title=None, sort=cat_order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("site:N", title=None, sort="-x"),
                color=alt.Color("AvgDiscount:Q", title="Avg discount (%)",
                                scale=alt.Scale(scheme="blues", domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("site:N", title="Site"),
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("AvgDiscount:Q", title="Avg discount (%)", format=".1f"),
                    alt.Tooltip("DiscCount:Q", title="Discounted SKUs (n)"),
                ],
            )
            .properties(height=340)
        )
        labels = (
            alt.Chart(heat_df)
            .mark_text(fontSize=11)
            .encode(
                x=alt.X("category:N", sort=cat_order),
                y="site:N",
                text="label:N",
                color=alt.value("#223"),
            )
        )
        st.altair_chart(heat + labels, use_container_width=True)

        # Compact pivot (hide zeros)
    #     pivot = (
    #         heat_df.pivot(index="site", columns="category", values="AvgDiscount")
    #         .reindex(columns=cat_order)
    #         .round(1)
    #     )
    #     pivot = pivot.where(pivot > 0)  # hide zeros/NaNs
    #     st.dataframe(pivot, use_container_width=True)
    # else:
    #     st.info("No discounted SKUs found to compute the (domain × category) average discount.")


def page_comparison(df: pd.DataFrame):
    st.header("Bally SKUs on Cettire")
    st.caption("Comprehensive Products analysis by AU E-Com")
    st.caption("Data scraped: 1–4 Oct 2025 (AUD). Internal use only.")

    dff, _ = comparison_filters(df)

    total = len(df)
    showing = len(dff)

    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown('<div class="app-wrap">', unsafe_allow_html=True)

    PER_PAGE = 50
    pages = max(1, math.ceil(showing / PER_PAGE))
    st.markdown('<div class="pagination-bar">', unsafe_allow_html=True)
    pager_cols = st.columns([1.2, 3, 3])
    with pager_cols[0]:
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)

    start = (page - 1) * PER_PAGE
    end = min(start + PER_PAGE, showing)
    if showing:
        display_text = f"Showing {start + 1:,}–{end:,} of {showing:,} comparisons"
    else:
        display_text = "Showing 0 comparisons"

    with pager_cols[1]:
        st.markdown(f"<div class='page-status'>{display_text}</div>", unsafe_allow_html=True)
    meta_label = "page" if pages == 1 else "pages"
    if showing == total:
        meta_text = f"{pages} {meta_label} • {total:,} SKUs"
    else:
        meta_text = f"{pages} {meta_label} • {showing:,} of {total:,} filtered"
    with pager_cols[2]:
        st.markdown(f"<div class='page-meta'>{meta_text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    page_rows = dff.iloc[start:end]

    st.markdown(f'<div class="count-pill">{display_text}</div>', unsafe_allow_html=True)

    for _, row in page_rows.iterrows():
        st.markdown(comp_card_html(row), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)





def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    df = load_data("match_final_gs.csv")  # <- your file name
    tab = st.sidebar.radio(
    "Go to",
    ["Analytics", "Comparison"],
    index=1,              # keep your default
    key="nav_go_to"   )    # <-- unique key fixes the duplicate-ID error)
    if tab == "Analytics":
        page_analytics(df)
    else:
        page_comparison(df)

if __name__ == "__main__":
    main()
print