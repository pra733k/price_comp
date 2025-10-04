# streamlit_app.py
# Bally products on Cettire – internal analytics & comparison UI
# Requirements: streamlit, pandas, numpy
# Data file expected: match_final.csv (AUD prices, schema provided by user)

import math
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# Page setup (Bally light style)
# ------------------------------
st.set_page_config(
    page_title="Bally products on Cettire",
    page_icon="👜",
    layout="wide",
)

# --- Bally-like light theme overrides (no dark UI) ---
st.markdown(
    """
    <style>
      :root {
        --bg: #FAFAFA;
        --card: #FFFFFF;
        --ink: #111111;
        --muted: #6B7280;
        --line: #E5E7EB;
        --accent: #CC0000;           /* price accent */
        --accent-2: #0F766E;         /* green-ish for cheaper badges if needed */
        --pill: #EEF2FF;
        --pill-text: #1F2937;
        --badge: #F3F4F6;
        --badge-text: #111111;
        --strike: #9CA3AF;
        --oos: #B91C1C;
      }
      .stApp { background: var(--bg); color: var(--ink); }
      .app-subtle { color: var(--muted); font-size: 13px; }

      /* section title spacing */
      .section { margin-top: 0.75rem; margin-bottom: .25rem; }

      /* anchor tabs header tweak */
      .stTabs [data-baseweb="tab"] { font-weight: 600; }

      /* Grid that holds comparison cards (two comps per row) */
      .comp-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(460px, 1fr));
        gap: 20px;
      }

      /* A comparison card houses the two tiles */
      .comp-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 2px 10px rgba(17,17,17,.04);
      }
      .comp-head { font-size: 13px; color: var(--muted); margin-bottom: .25rem; }

      /* Two product tiles side-by-side */
      .tiles {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
      }

      /* Entire tile is clickable */
      .tile-link { text-decoration: none; color: inherit; display: block; }
      .tile {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        transition: transform .12s ease, box-shadow .12s ease;
      }
      .tile:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(17,17,17,.06); }

      /* preserved aspect ratio like Bally cards */
      .img-wrap {
        width: 100%;
        aspect-ratio: 4/5;
        background: #F5F5F5;
        display: grid;
        place-items: center;
      }
      .img-wrap img {
        max-width: 92%;
        max-height: 92%;
        object-fit: contain;
        display: block;
      }

      .panel-body { padding: 12px 14px 14px 14px; }
      .site-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
      .title { font-size: 16px; line-height: 1.25; font-weight: 600; min-height: 40px; }
      .badges { display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
      .badge {
        background: var(--badge);
        color: var(--badge-text);
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 12px;
        border: 1px solid var(--line);
      }
      .badge.oos {
        background: #FEE2E2; color: var(--oos); border-color: #FECACA;
      }
      .price-line { margin-top: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
      .strike { color: var(--strike); text-decoration: line-through; }
      .price-accent { color: var(--accent); font-weight: 700; }
      .pct-pill {
        background: var(--pill); color: var(--pill-text);
        border-radius: 999px; font-size: 12px; padding: 2px 8px;
      }

      .diff-bar { margin-top: 10px; font-size: 13px; color: var(--muted); }
      .diff-pos { color: var(--accent); font-weight: 600; }
      .diff-neg { color: var(--accent-2); font-weight: 600; }

      /* hyperlink tables in analytics */
      .tbl-note { font-size: 12px; color: var(--muted); margin-top: .25rem; }
      .kpi { font-size: 26px; font-weight: 700; }
      .kpi-sub { font-size: 13px; color: var(--muted); }

      /* pagination strip */
      .pager { display: flex; gap: 8px; align-items: center; }
      .pager input { width: 70px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Utilities
# ------------------------------
def to_number(x) -> Optional[float]:
    """Robust parse of AUD numeric fields (already in AUD). Returns float or np.nan."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "na" or s.lower() == "none":
        return np.nan
    # remove currency labels and commas/spaces
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def final_price(row: pd.Series, side: str) -> Optional[float]:
    """Final price = sale if present else retail. Returns np.nan if both missing or OOS (match only)."""
    if side == "m":
        # Out of stock tag on match side ⇒ no final price
        tag = str(row.get("m_season_tag") or "").lower()
        if "out of stock" in tag or "out-of-stock" in tag or "oos" in tag:
            return np.nan
    sale = to_number(row.get(f"{side}_sale_price"))
    retail = to_number(row.get(f"{side}_retail_price"))
    if not np.isnan(sale):
        return sale
    return retail if not np.isnan(retail) else np.nan

def pct_discount(retail, sale) -> Optional[float]:
    r = to_number(retail); s = to_number(sale)
    if np.isnan(r) or np.isnan(s) or r <= 0 or s >= r:
        return np.nan
    return round((r - s) * 100.0 / r, 2)

def fmt_money(x) -> str:
    v = to_number(x)
    if np.isnan(v):
        return "—"
    return f"AUD$ {v:,.2f}"

def linkify(url: str, text: str) -> str:
    if not url or str(url).strip().lower() in ("nan", "na", "none"):
        return text
    safe = str(url).replace('"', "%22")
    return f'<a href="{safe}" target="_blank" rel="noopener noreferrer">{text}</a>'

def clean_domain(x) -> str:
    s = str(x or "").strip()
    if s.lower() in ("", "nan", "na", "none"):
        return "NA"
    return s

def site_priority(domain: str) -> int:
    d = clean_domain(domain).lower()
    if d == "www.bally.com.au": return 0
    if d.startswith("www.bally."): return 1
    if d.startswith("bally"): return 1
    if d == "www.farfetch.com": return 2
    if d == "na": return 99
    if d == "na": return 99
    if d == "nan": return 99
    if d == "": return 99
    return 3

# ------------------------------
# Data loading & preparation
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    # Normalize columns that must exist even if missing
    for col in [
        "c_link","domain","category","c_title","c_retail_price","c_sale_price","c_image-src",
        "c_season_tag","c_product_url","c_product_id","matchlink","m_title","m_retail_price",
        "m_sale_price","m_image-src","m_product_id","m_season_tag"
    ]:
        if col not in df.columns:
            df[col] = ""

    # Clean types
    df["domain"] = df["domain"].apply(clean_domain)
    df["c_final"] = df.apply(lambda r: final_price(r, "c"), axis=1)
    df["m_final"] = df.apply(lambda r: final_price(r, "m"), axis=1)
    df["has_match"] = (df["domain"].str.lower() != "na") & (df["matchlink"].astype(str).str.len() > 2)

    # Numeric price columns
    for p in ["c_retail_price","c_sale_price","m_retail_price","m_sale_price","c_final","m_final"]:
        df[p] = df[p].apply(to_number)

    # Differences (only when both finals present)
    both = (~df["c_final"].isna()) & (~df["m_final"].isna())
    df["diff"] = np.where(both, df["c_final"] - df["m_final"], np.nan)

    # Discounts (per-row)
    df["c_pct_disc"] = df.apply(lambda r: pct_discount(r["c_retail_price"], r["c_sale_price"]), axis=1)
    df["m_pct_disc"] = df.apply(lambda r: pct_discount(r["m_retail_price"], r["m_sale_price"]), axis=1)

    # Simple badges
    df["c_tag"] = df["c_season_tag"].fillna("").astype(str)
    df["m_tag"] = df["m_season_tag"].fillna("").astype(str)

    # Status
    df["m_is_oos"] = df["m_tag"].str.lower().str.contains("out of stock")
    df["site_priority"] = df["domain"].apply(site_priority)

    # Season universe for filters
    seasons = (
        pd.Series(pd.concat([df["c_tag"], df["m_tag"]], ignore_index=True).unique())
        .dropna().astype(str).str.strip()
    )
    seasons = seasons[seasons != ""].tolist()
    return df, sorted(set(seasons))

# ------------------------------
# Analytics helpers
# ------------------------------
def group_site_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    mask = (~d["c_final"].isna()) & (~d["m_final"].isna()) & (~d["m_is_oos"])
    g = (
        d[mask]
        .groupby("domain", dropna=False, as_index=False)
        .agg(
            Rows=("diff","count"),
            Average_Difference=("diff","mean"),
            Median_Difference=("diff","median"),
            Pct_Cheaper_Cettire=("diff", lambda s: 100.0 * np.mean(d.loc[s.index, "c_final"] < d.loc[s.index, "m_final"]))
        )
        .sort_values("Rows", ascending=False)
    )
    g["Average_Difference"] = g["Average_Difference"].round(2)
    g["Median_Difference"] = g["Median_Difference"].round(2)
    g["Pct_Cheaper_Cettire"] = g["Pct_Cheaper_Cettire"].round(1)
    return g

def bally_au_by_category(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d[d["domain"].str.lower() == "www.bally.com.au"]
    mask = (~d["c_final"].isna()) & (~d["m_final"].isna()) & (~d["m_is_oos"])
    g = (
        d[mask]
        .groupby("category", as_index=False)
        .agg(
            Rows=("diff","count"),
            Average_Difference=("diff","mean"),
            Median_Difference=("diff","median"),
            Pct_Cheaper_Cettire=("diff", lambda s: 100.0 * np.mean(d.loc[s.index, "c_final"] < d.loc[s.index, "m_final"]))
        )
        .sort_values("Rows", ascending=False)
    )
    for col in ["Average_Difference","Median_Difference","Pct_Cheaper_Cettire"]:
        g[col] = g[col].round(2)
    return g

def top_discrepancies(df: pd.DataFrame, n=10) -> pd.DataFrame:
    d = df.copy()
    mask = (d["domain"].str.lower() == "www.bally.com.au") & (~d["m_is_oos"]) & (~d["diff"].isna())
    d = d[mask].copy()
    d["absdiff"] = d["diff"].abs()
    d = d.sort_values("absdiff", ascending=False).head(n)
    out = pd.DataFrame({
        "category": d["category"],
        "domain": d["domain"],
        "c_title": d["c_title"],
        "m_title": d["m_title"],
        "Cettire final (AUD)": d["c_final"].map(lambda x: f"{x:,.2f}"),
        "Match final (AUD)": d["m_final"].map(lambda x: f"{x:,.2f}"),
        "Diff (Cettire - Match)": d["diff"].map(lambda x: f"{x:,.2f}"),
        "Cettire URL": d["c_link"].apply(lambda u: linkify(u, "Open")),
        "Match URL": d["matchlink"].apply(lambda u: linkify(u, "Open")),
    })
    return out

def discounting_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Percent rows with sale < retail (both present). Match excludes OOS.
    d["c_disc_row"] = (~d["c_sale_price"].isna()) & (~d["c_retail_price"].isna()) & (d["c_sale_price"].apply(to_number) < d["c_retail_price"].apply(to_number))
    d["m_disc_row"] = (~d["m_sale_price"].isna()) & (~d["m_retail_price"].isna()) & (~d["m_is_oos"]) & (d["m_sale_price"].apply(to_number) < d["m_retail_price"].apply(to_number))

    g = (
        d.groupby("domain", dropna=False, as_index=False)
        .agg(
            Rows=("domain","count"),
            Pct_Cettire_Discount=("c_disc_row", lambda s: 100.0 * s.mean() if len(s) else 0.0),
            Pct_Match_Discount=("m_disc_row", lambda s: 100.0 * s.mean() if len(s) else 0.0),
        )
        .sort_values("Rows", ascending=False)
    )
    g["Pct_Cettire_Discount"] = g["Pct_Cettire_Discount"].round(2)
    g["Pct_Match_Discount"] = g["Pct_Match_Discount"].round(2)
    return g

# ------------------------------
# Tile / card rendering
# ------------------------------
def tile_html(
    title: str,
    img: str,
    site_label: str,
    season_tag: str,
    retail: Optional[float],
    sale: Optional[float],
    final_price_val: Optional[float],
    url: str,
    is_oos: bool = False,
) -> str:
    retail_s = fmt_money(retail)
    sale_s = fmt_money(sale)
    final_s = fmt_money(final_price_val)
    disc = pct_discount(retail, sale)
    badge = ""
    if season_tag:
        tag = season_tag.strip()
        badge = f'<span class="badge">{tag}</span>'

    if is_oos:
        badge += '<span class="badge oos">OOS</span>'

    # price block
    price_bits = []
    if is_oos:
        price_bits.append('<span class="price-accent">—</span>')
    else:
        # show strike only if sale < retail
        if not np.isnan(to_number(sale)) and not np.isnan(to_number(retail)) and to_number(sale) < to_number(retail):
            price_bits.append(f'<span class="strike">{retail_s}</span>')
            price_bits.append(f'<span class="price-accent">{sale_s}</span>')
            if disc is not None and not np.isnan(disc):
                price_bits.append(f'<span class="pct-pill">-{disc:.0f}%</span>')
        else:
            # just final
            price_bits.append(f'<span class="price-accent">{final_s}</span>')

    return f"""
      <a class="tile-link" href="{url or '#'}" target="_blank" rel="noopener noreferrer">
        <div class="tile">
          <div class="img-wrap">
            <img src="{img or ''}" alt="{(title or '').replace('"','')}" loading="lazy" />
          </div>
          <div class="panel-body">
            <div class="site-label">{site_label}</div>
            <div class="title">{(title or '').strip()}</div>
            <div class="badges">{badge}</div>
            <div class="price-line">{' '.join(price_bits)}</div>
          </div>
        </div>
      </a>
    """

def comp_card_html(row: pd.Series) -> str:
    # Left (Cettire)
    c_tile = tile_html(
        title=row.get("c_title"),
        img=row.get("c_image-src"),
        site_label="Cettire",
        season_tag=row.get("c_tag"),
        retail=row.get("c_retail_price"),
        sale=row.get("c_sale_price"),
        final_price_val=row.get("c_final"),
        url=row.get("c_link") or row.get("c_product_url"),
        is_oos=False,
    )
    # Right (Match)
    is_oos = bool(row.get("m_is_oos"))
    m_title = row.get("m_title") or ("No match" if not row.get("has_match") else row.get("m_title"))
    m_img = row.get("m_image-src") if row.get("has_match") else ""
    m_url = row.get("matchlink") if row.get("has_match") else "#"
    m_tile = tile_html(
        title=m_title,
        img=m_img,
        site_label=row.get("domain"),
        season_tag=row.get("m_tag"),
        retail=row.get("m_retail_price"),
        sale=row.get("m_sale_price"),
        final_price_val=row.get("m_final"),
        url=m_url,
        is_oos=is_oos,
    )

    # Diff label
    diff = row.get("diff")
    if diff is None or np.isnan(diff):
        diff_label = '<span class="app-subtle">Diff: —</span>'
    else:
        cls = "diff-pos" if diff > 0 else "diff-neg"
        diff_label = f'Diff (Cettire − Match): <span class="{cls}">AUD$ {diff:,.2f}</span>'

    return f"""
      <div class="comp-card">
        <div class="tiles">
          {c_tile}
          {m_tile}
        </div>
        <div class="diff-bar">{diff_label}</div>
      </div>
    """

# ------------------------------
# App body
# ------------------------------
st.title("Bally products on Cettire")
st.caption("Comprehensive Products analysis by AU E-Com")
st.caption("Data scraped: 1–4 Oct 2025 (AUD). Internal use only.")

tabs = st.tabs(["📊 Analytics", "🧾 Comparison"])

# Load data
df, season_universe = load_data("match_final.csv")

# ========= ANALYTICS =========
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total rows", f"{len(df):,}")
    with c2:
        st.metric("Cettire SKUs", f"{len(df):,}")
    with c3:
        st.metric("Matched rows", f"{int(df['has_match'].sum()):,}")
    with c4:
        st.metric("OOS (match)", f"{int(df['m_is_oos'].sum()):,}")

    st.markdown("### Site-wise comparison")
    st.caption("Average Difference = mean(Cettire final − Match final) using rows with both prices and non-OOS. Positive means Cettire is more expensive; negative means cheaper.")

    g_site = group_site_summary(df)
    st.dataframe(g_site, use_container_width=True, hide_index=True)
    st.download_button("Download site summary (CSV)", g_site.to_csv(index=False).encode("utf-8"), "site_summary.csv", "text/csv")

    st.markdown("### Bally AU – category difference (select site)")
    st.caption("Per-category average and median difference for **www.bally.com.au**. Numeric panels exclude OOS and rows without both final prices.")
    g_cat = bally_au_by_category(df)
    st.dataframe(g_cat, use_container_width=True, hide_index=True)
    st.download_button("Download bally.com.au category summary (CSV)", g_cat.to_csv(index=False).encode("utf-8"), "bally_au_category_summary.csv", "text/csv")

    st.markdown("### Top price discrepancies (Bally AU match)")
    st.caption("Largest absolute differences where the matched site is **www.bally.com.au**, excluding OOS.")
    topn = top_discrepancies(df, n=10)
    # render as clickable HTML table (escape disabled)
    st.markdown(topn.to_html(index=False, escape=False), unsafe_allow_html=True)
    st.download_button("Download discrepancies (CSV)", topn.drop(columns=["Cettire URL","Match URL"]).to_csv(index=False).encode("utf-8"), "top_discrepancies.csv", "text/csv")

    st.markdown("### Discounting summary (by site)")
    st.caption("Cettire discount = % where sale < retail on Cettire; Match discount = % where sale < retail on matched site (excludes OOS).")
    disc = discounting_summary(df)
    st.dataframe(disc, use_container_width=True, hide_index=True)
    st.download_button("Download discounting (CSV)", disc.to_csv(index=False).encode("utf-8"), "discounting_by_site.csv", "text/csv")

# ========= COMPARISON =========
with tabs[1]:
    st.markdown("#### Products")

    # --- Filters sidebar-like block ---
    fcol, pcol, _ = st.columns([1.1, 3, 0.1])

    with fcol:
        st.markdown("**Filters**")

        # Category
        cats = sorted([c for c in df["category"].dropna().astype(str).unique().tolist() if c.strip() != ""])
        sel_cats = st.multiselect("Category", options=cats, default=cats)

        # Matched site
        sites = sorted(df["domain"].dropna().astype(str).unique().tolist(), key=site_priority)
        sel_sites = st.multiselect("Matched site", options=sites, default=sites)

        # Season tag (either side)
        sel_seasons = st.multiselect("Season tag (either side)", options=sorted(season_universe), default=sorted(season_universe))

        include_na = st.checkbox("Include NA / OOS", value=True)

        st.markdown("**Price filter (Cettire final)**")
        bucket = st.selectbox("Price bucket", ["All","0–500","500–1,000","1,000–2,000","2,000–4,000","4,000+"])
        manual_min = st.text_input("Manual min (AUD)", "", placeholder="leave blank")
        manual_max = st.text_input("Manual max (AUD)", "", placeholder="leave blank")

        sort_mode = st.selectbox("Sort", [
            "Max price diff (abs desc)",
            "Cettire final (low → high)",
            "Cettire final (high → low)",
            "Match final (low → high)",
            "Match final (high → low)",
            "Site priority, then max diff",
        ])

    # --- Apply filters to df ---
    d = df.copy()

    # Category
    d = d[d["category"].astype(str).isin(sel_cats)]

    # Site
    d = d[d["domain"].astype(str).isin(sel_sites)]

    # Season (either side)
    if sel_seasons:
        has_tag = (
            d["c_tag"].astype(str).isin(sel_seasons) |
            d["m_tag"].astype(str).isin(sel_seasons)
        )
        d = d[has_tag]

    # NA / OOS toggle
    if not include_na:
        d = d[d["has_match"] & (~d["m_is_oos"])]

    # Price bucket (Cettire final)
    min_b, max_b = None, None
    if bucket != "All":
        if bucket == "0–500":        min_b, max_b = 0, 500
        elif bucket == "500–1,000":  min_b, max_b = 500, 1000
        elif bucket == "1,000–2,000":min_b, max_b = 1000, 2000
        elif bucket == "2,000–4,000":min_b, max_b = 2000, 4000
        elif bucket == "4,000+":     min_b, max_b = 4000, None

    # Manual min/max override if numbers provided
    try:
        if manual_min.strip():
            min_b = float(re.sub(r"[^\d.]", "", manual_min))
    except Exception:
        pass
    try:
        if manual_max.strip():
            max_b = float(re.sub(r"[^\d.]", "", manual_max))
    except Exception:
        pass

    if min_b is not None:
        d = d[(~d["c_final"].isna()) & (d["c_final"] >= min_b)]
    if max_b is not None:
        d = d[(~d["c_final"].isna()) & (d["c_final"] <= max_b)]

    # Sorting
    if sort_mode == "Max price diff (abs desc)":
        d = d.sort_values(by=["diff"], key=lambda s: s.abs(), ascending=False)
    elif sort_mode == "Cettire final (low → high)":
        d = d.sort_values(by=["c_final"], ascending=True, na_position="last")
    elif sort_mode == "Cettire final (high → low)":
        d = d.sort_values(by=["c_final"], ascending=False, na_position="last")
    elif sort_mode == "Match final (low → high)":
        d = d.sort_values(by=["m_final"], ascending=True, na_position="last")
    elif sort_mode == "Match final (high → low)":
        d = d.sort_values(by=["m_final"], ascending=False, na_position="last")
    else:  # Site priority, then max diff
        d = d.sort_values(by=["site_priority","diff"], key=lambda s: s if s.name!="diff" else s.abs(), ascending=[True, False])

    total_results = len(d)
    per_page = 25   # 25 comparisons = 50 tiles per page
    total_pages = max(1, math.ceil(total_results / per_page))

    # --- Pagination controls (top) ---
    with pcol:
        left, mid, right = st.columns([1,3,2])
        with left:
            st.markdown("**Showing**")
            # numeric page input (1-indexed)
            pg = st.number_input("", min_value=1, max_value=total_pages, value=1, step=1, label_visibility="collapsed")
        with mid:
            st.markdown(f"<div class='app-subtle'>({total_pages} pages)</div>", unsafe_allow_html=True)
        with right:
            st.markdown(f"<div class='app-subtle' style='text-align:right'>{total_results:,} results</div>", unsafe_allow_html=True)

        start = (pg - 1) * per_page
        end = start + per_page
        page_rows = d.iloc[start:end].copy()

        # Render grid
        st.markdown("<div class='comp-grid'>", unsafe_allow_html=True)
        for _, r in page_rows.iterrows():
            st.markdown(comp_card_html(r), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
