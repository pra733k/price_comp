import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Bally Product Analysis",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Bally-inspired CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #FAFAFA;
    }
    
    .bally-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: white;
        border-bottom: 1px solid #E5E5E5;
        margin-bottom: 2rem;
    }
    
    .bally-logo {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 0.3em;
        color: #000000;
        margin-bottom: 0.5rem;
    }
    
    .bally-subtitle {
        font-size: 0.95rem;
        color: #666666;
        font-weight: 400;
        margin-bottom: 0.3rem;
    }
    
    .bally-caption {
        font-size: 0.75rem;
        color: #999999;
        font-weight: 300;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: white;
        padding: 0;
        border-bottom: 1px solid #E5E5E5;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: white;
        border: none;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        color: #666666;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F5F5F5;
        color: #000000;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #000000;
        border-bottom: 2px solid #000000;
    }
    
    .result-count {
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 1.5rem;
        font-weight: 400;
        text-align: center;
    }
    
    .stButton > button {
        background-color: #000000;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        border-radius: 2px;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #333333;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #000000;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #000000;
    }
    
    .explanation-text {
        font-size: 0.85rem;
        color: #666666;
        font-style: italic;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    /* Product card styling */
    div[data-testid="column"] {
        padding: 0.5rem;
    }
    
    .product-container {
        background: white;
        border: 1px solid #E5E5E5;
        border-radius: 2px;
        padding: 1rem;
        height: 100%;
        transition: box-shadow 0.2s;
    }
    
    .product-container:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .site-label-header {
        font-size: 0.75rem;
        color: #666666;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    .season-badge-inline {
        display: inline-block;
        background: white;
        padding: 3px 8px;
        font-size: 0.7rem;
        font-weight: 500;
        border: 1px solid #E5E5E5;
        border-radius: 2px;
        letter-spacing: 0.03em;
        margin-top: 0.5rem;
    }
    
    .oos-badge-inline {
        display: inline-block;
        background: #000000;
        color: white;
        padding: 3px 8px;
        font-size: 0.7rem;
        font-weight: 500;
        border-radius: 2px;
        letter-spacing: 0.03em;
        margin-top: 0.5rem;
    }
    
    .price-display {
        text-align: center;
        margin-top: 0.75rem;
    }
    
    .retail-price-text {
        font-size: 0.85rem;
        color: #999999;
        text-decoration: line-through;
        margin-right: 6px;
    }
    
    .sale-price-text {
        font-size: 1rem;
        font-weight: 600;
        color: #C41E3A;
    }
    
    .final-price-text {
        font-size: 1rem;
        font-weight: 600;
        color: #000000;
    }
    
    .discount-badge-inline {
        display: inline-block;
        background: #C41E3A;
        color: white;
        padding: 3px 8px;
        font-size: 0.7rem;
        font-weight: 600;
        border-radius: 2px;
        margin-left: 6px;
    }
    
    .tile-separator {
        margin: 2rem 0;
        border-top: 1px solid #E5E5E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and preprocess the CSV data"""
    df = pd.read_csv('match_final.csv')
    
    # Calculate final prices
    df['c_final_price'] = df['c_sale_price'].fillna(df['c_retail_price'])
    df['m_final_price'] = df['m_sale_price'].fillna(df['m_retail_price'])
    
    # Calculate price difference
    df['price_diff'] = df.apply(
        lambda row: row['c_final_price'] - row['m_final_price'] 
        if pd.notna(row['c_final_price']) and pd.notna(row['m_final_price']) 
        else np.nan, 
        axis=1
    )
    
    # Normalize season tags
    season_mapping = {
        'new season': 'New Season',
        'full price': 'Full Price',
        'online exclusive': 'Online Exclusive',
        'sale': 'Sale',
        'final sale': 'Final Sale',
        'out of stock': 'Out Of Stock',
        'pre-fall': 'Pre-fall',
        'na': 'NA'
    }
    
    df['c_season_tag'] = df['c_season_tag'].fillna('No tag').astype(str).str.lower().str.strip()
    df['m_season_tag'] = df['m_season_tag'].fillna('No tag').astype(str).str.lower().str.strip()
    
    df['c_season_tag'] = df['c_season_tag'].replace(season_mapping)
    df['m_season_tag'] = df['m_season_tag'].replace(season_mapping)
    
    df['c_season_tag'] = df['c_season_tag'].str.title()
    df['m_season_tag'] = df['m_season_tag'].str.title()
    
    # Flag OOS rows
    df['is_oos'] = df['m_season_tag'] == 'Out Of Stock'
    
    # Domain grouping
    def get_domain_group(domain):
        if pd.isna(domain):
            return 'Other'
        domain = str(domain).lower()
        if 'bally.com.au' in domain:
            return 'Bally AU'
        elif 'bally.' in domain:
            return 'Bally International'
        elif 'farfetch' in domain:
            return 'Farfetch'
        else:
            return 'Other'
    
    df['domain_group'] = df['domain'].apply(get_domain_group)
    
    # Domain priority for sorting
    domain_priority = {
        'Bally AU': 1,
        'Bally International': 2,
        'Farfetch': 3,
        'Other': 4
    }
    df['domain_priority'] = df['domain_group'].map(domain_priority)
    
    # Season priority for sorting
    season_priority = {
        'New Season': 1,
        'Full Price': 2,
        'Online Exclusive': 3,
        'Sale': 4,
        'Final Sale': 5,
        'Pre-Fall': 6,
        'No Tag': 7,
        'Na': 8,
        'Out Of Stock': 9
    }
    df['c_season_priority'] = df['c_season_tag'].map(season_priority).fillna(10)
    
    # Calculate discount percentages
    df['c_discount_pct'] = df.apply(
        lambda row: ((row['c_retail_price'] - row['c_sale_price']) / row['c_retail_price'] * 100)
        if pd.notna(row['c_retail_price']) and pd.notna(row['c_sale_price']) and row['c_retail_price'] > 0
        else np.nan,
        axis=1
    )
    
    df['m_discount_pct'] = df.apply(
        lambda row: ((row['m_retail_price'] - row['m_sale_price']) / row['m_retail_price'] * 100)
        if pd.notna(row['m_retail_price']) and pd.notna(row['m_sale_price']) and row['m_retail_price'] > 0
        else np.nan,
        axis=1
    )
    
    return df


def render_header():
    """Render the Bally-styled header"""
    st.markdown("""
        <div class="bally-header">
            <div class="bally-logo">BALLY</div>
            <div class="bally-subtitle">Comprehensive Products analysis by AU E-Com</div>
            <div class="bally-caption">Data scraped: 1–4 Oct 2025 (AUD). Internal use only.</div>
        </div>
    """, unsafe_allow_html=True)


def render_product_card(row, side='left'):
    """Render a single product card using Streamlit columns"""
    
    if side == 'left':
        is_oos = row['is_oos']
        season = row['c_season_tag'] if pd.notna(row['c_season_tag']) and row['c_season_tag'] not in ['No Tag', 'No tag'] else ''
        title = str(row['c_title']) if pd.notna(row['c_title']) else 'N/A'
        img = str(row['c_image-src']) if pd.notna(row['c_image-src']) else None
        url = str(row['c_product_url']) if pd.notna(row['c_product_url']) else '#'
        retail = row.get('c_retail_price')
        sale = row.get('c_sale_price')
        final = row.get('c_final_price')
        discount = row.get('c_discount_pct')
        label = "CETTIRE"
    else:
        is_oos = row['m_season_tag'] == 'Out Of Stock'
        season = row['m_season_tag'] if pd.notna(row['m_season_tag']) and row['m_season_tag'] not in ['No Tag', 'No tag', 'Na'] else ''
        title = str(row['m_title']) if pd.notna(row['m_title']) else 'N/A'
        img = str(row['m_image-src']) if pd.notna(row['m_image-src']) else None
        url = str(row['matchlink']) if pd.notna(row['matchlink']) else '#'
        retail = row.get('m_retail_price')
        sale = row.get('m_sale_price')
        final = row.get('m_final_price')
        discount = row.get('m_discount_pct')
        label = str(row['domain']).upper() if pd.notna(row['domain']) else 'N/A'
    
    with st.container():
        st.markdown(f'<div class="product-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="site-label-header">{label}</div>', unsafe_allow_html=True)
        
        # Image
        if img and img != 'None':
            try:
                st.image(img, use_container_width=True)
            except:
                st.markdown('<div style="height: 200px; background: #f0f0f0; display: flex; align-items: center; justify-content: center; color: #999;">No Image</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="height: 200px; background: #f0f0f0; display: flex; align-items: center; justify-content: center; color: #999;">No Image</div>', unsafe_allow_html=True)
        
        # Season badge
        if is_oos:
            st.markdown('<div style="text-align: center;"><span class="oos-badge-inline">OUT OF STOCK</span></div>', unsafe_allow_html=True)
        elif season:
            st.markdown(f'<div style="text-align: center;"><span class="season-badge-inline">{season}</span></div>', unsafe_allow_html=True)
        
        # Title with link
        st.markdown(f'<div style="text-align: center; margin: 0.75rem 0; min-height: 3rem;"><a href="{url}" target="_blank" style="color: #000; text-decoration: none; font-size: 0.9rem; font-weight: 500;">{title}</a></div>', unsafe_allow_html=True)
        
        # Prices - only show if not OOS
        if not is_oos:
            price_html = '<div class="price-display">'
            if pd.notna(sale) and pd.notna(retail):
                discount_badge = f'<span class="discount-badge-inline">-{int(discount)}%</span>' if pd.notna(discount) else ''
                price_html += f'<span class="retail-price-text">AUD ${retail:.2f}</span><span class="sale-price-text">AUD ${sale:.2f}</span>{discount_badge}'
            elif pd.notna(final):
                price_html += f'<span class="final-price-text">AUD ${final:.2f}</span>'
            price_html += '</div>'
            st.markdown(price_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def comparison_page(df):
    """Render the Comparison page"""
    
    st.markdown('<h2 style="text-align: center; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 2rem;">BALLY PRODUCTS ON CETTIRE</h2>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div style="font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem; color: #000000;">Filters</div>', unsafe_allow_html=True)
        
        categories = sorted([c for c in df['category'].dropna().unique() if str(c) != 'nan'])
        selected_categories = st.multiselect('Category', categories, default=[])
        
        domains = sorted([d for d in df['domain'].dropna().unique() if str(d) != 'nan'])
        selected_domains = st.multiselect('Site / Domain', domains, default=[])
        
        all_seasons = sorted(set(df['c_season_tag'].unique()) | set(df['m_season_tag'].unique()))
        all_seasons = [s for s in all_seasons if s not in ['No Tag', 'Na', 'nan', 'No tag'] and str(s) != 'nan']
        selected_seasons = st.multiselect('Season Tag', all_seasons, default=[])
        
        include_na_oos = st.checkbox('Include NA/OOS', value=True)
        
        st.markdown('**Cettire Price Range**')
        price_buckets = [
            'All',
            '0 – 250',
            '250 – 500',
            '500 – 1,000',
            '1,000 – 1,500',
            '1,500 – 3,000',
            '3,000+'
        ]
        selected_bucket = st.selectbox('Select Range', price_buckets, index=0)
        
        use_custom = st.checkbox('Use custom range')
        min_price = 0
        max_price = 10000
        if use_custom:
            col1, col2 = st.columns(2)
            with col1:
            filter_by = st.selectbox('Filter by', ['All', 'Category', 'Site'])
        
        with col2:
            if filter_by == 'Category':
                filter_value = st.selectbox('Select', sorted(df_price_valid['category'].unique()))
                filtered_dist = df_price_valid[df_price_valid['category'] == filter_value]
            elif filter_by == 'Site':
                filter_value = st.selectbox('Select', sorted(df_price_valid['domain_group'].unique()))
                filtered_dist = df_price_valid[df_price_valid['domain_group'] == filter_value]
            else:
                filtered_dist = df_price_valid
        
        fig_hist = px.histogram(
            filtered_dist,
            x='price_diff',
            nbins=50,
            title='Price Difference Distribution',
            labels={'price_diff': 'Price Difference (AUD)'},
            color_discrete_sequence=['#000000']
        )
        fig_hist.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', size=12),
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        fig_box = px.box(
            filtered_dist,
            y='price_diff',
            title='Price Difference Boxplot',
            labels={'price_diff': 'Price Difference (AUD)'},
            color_discrete_sequence=['#000000']
        )
        fig_box.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', size=12),
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("No valid price comparison data available.")
    
    # TOP 10 PRICE DISCREPANCIES
    st.markdown('<div class="section-header">Top 10 Price Discrepancies</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Products with the largest absolute price differences (positive or negative).</div>', unsafe_allow_html=True)
    
    if not df_price_valid.empty:
        df_price_valid_copy = df_price_valid.copy()
        df_price_valid_copy['abs_price_diff'] = df_price_valid_copy['price_diff'].abs()
        top_10 = df_price_valid_copy.nlargest(10, 'abs_price_diff')[
            ['c_image-src', 'c_title', 'c_final_price', 'm_final_price', 'price_diff', 'domain', 'c_product_url', 'matchlink']
        ].copy()
        
        for idx, row in top_10.iterrows():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                img_url = row['c_image-src'] if pd.notna(row['c_image-src']) else ''
                if img_url:
                    try:
                        st.image(img_url, width=100)
                    except:
                        st.write("Image unavailable")
            
            with col2:
                st.markdown(f"**{row['c_title']}**")
                st.markdown(f"Cettire: AUD ${row['c_final_price']:.2f} | {row['domain']}: AUD ${row['m_final_price']:.2f}")
                diff_color = "green" if row['price_diff'] < 0 else "red"
                st.markdown(f"**Price Difference: <span style='color:{diff_color}'>AUD ${row['price_diff']:.2f}</span>**", unsafe_allow_html=True)
                st.markdown(f"[Cettire Link]({row['c_product_url']}) | [Match Link]({row['matchlink']})")
            
            st.markdown("---")
    else:
        st.warning("No valid price comparison data available.")
    
    # OUT OF STOCK & NA SUMMARY
    st.markdown('<div class="section-header">Out of Stock & NA Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Count of products that are out of stock or have no matching data. These are excluded from price difference calculations.</div>', unsafe_allow_html=True)
    
    oos_count = df[df['is_oos']].shape[0]
    na_count = df[df['price_diff'].isna()].shape[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 2px; border: 1px solid #E5E5E5; margin-bottom: 1.5rem;">
            <div style="font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #000000; margin-bottom: 0.5rem;">OUT OF STOCK</div>
            <div style="font-size: 2rem; font-weight: 700; color: #000000; margin-bottom: 0.25rem;">{oos_count}</div>
            <div style="font-size: 0.8rem; color: #666666;">Products marked as OOS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 2px; border: 1px solid #E5E5E5; margin-bottom: 1.5rem;">
            <div style="font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #000000; margin-bottom: 0.5rem;">MISSING MATCH DATA</div>
            <div style="font-size: 2rem; font-weight: 700; color: #000000; margin-bottom: 0.25rem;">{na_count}</div>
            <div style="font-size: 0.8rem; color: #666666;">Products with no price comparison</div>
        </div>
        """, unsafe_allow_html=True)
    
    # OOS by Category
    oos_by_category = df[df['is_oos']].groupby('category').size().reset_index(name='Count')
    if not oos_by_category.empty:
        st.markdown("**OOS by Category**")
        st.dataframe(oos_by_category, use_container_width=True)
    else:
        st.info("No out-of-stock products by category.")
    
    # OOS by Site
    oos_by_site = df[df['is_oos']].groupby('domain_group').size().reset_index(name='Count')
    if not oos_by_site.empty:
        st.markdown("**OOS by Site**")
        st.dataframe(oos_by_site, use_container_width=True)
    else:
        st.info("No out-of-stock products by site.")
    
    # NA by Category
    na_by_category = df[df['price_diff'].isna()].groupby('category').size().reset_index(name='Count')
    if not na_by_category.empty:
        st.markdown("**NA/Missing Match by Category**")
        st.dataframe(na_by_category, use_container_width=True)
    else:
        st.info("No missing match data by category.")
    
    # NA by Site
    na_by_site = df[df['price_diff'].isna()].groupby('domain_group').size().reset_index(name='Count')
    if not na_by_site.empty:
        st.markdown("**NA/Missing Match by Site**")
        st.dataframe(na_by_site, use_container_width=True)
    else:
        st.info("No missing match data by site.")
    
    # DOWNLOAD DATA
    st.markdown('<div class="section-header">Download Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Export full dataset or filtered subsets for external analysis.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Dataset",
            data=csv_full,
            file_name=f"bally_analysis_full_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_filtered = df_price_valid.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Dataset (Price Valid)",
            data=csv_filtered,
            file_name=f"bally_analysis_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def main():
    """Main application"""
    render_header()
    
    df = load_data()
    
    if df is None or df.empty:
        st.error("Failed to load data. Please check that 'match_final.csv' exists in the same directory as this script.")
        st.stop()
    
    tab1, tab2 = st.tabs(["ANALYTICS", "COMPARISON"])
    
    with tab1:
        analytics_page(df)
    
    with tab2:
        comparison_page(df)


if __name__ == "__main__":
    main()1:
                min_price = st.number_input('Min', min_value=0, value=0)
            with col2:
                max_price = st.number_input('Max', min_value=0, value=10000)
        
        st.markdown('**Sort By**')
        sort_options = [
            'Max price diff (desc)',
            'Cettire price (desc)',
            'Cettire price (asc)',
            'Bally AU price (desc)',
            'Bally AU price (asc)',
            'Title A-Z',
            'Season priority',
            'Site priority'
        ]
        selected_sort = st.selectbox('Sort', sort_options, index=0)
    
    filtered_df = df.copy()
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    if selected_domains:
        filtered_df = filtered_df[filtered_df['domain'].isin(selected_domains)]
    
    if selected_seasons:
        filtered_df = filtered_df[
            (filtered_df['c_season_tag'].isin(selected_seasons)) | 
            (filtered_df['m_season_tag'].isin(selected_seasons))
        ]
    
    if not include_na_oos:
        filtered_df = filtered_df[~filtered_df['is_oos']]
        filtered_df = filtered_df[filtered_df['price_diff'].notna()]
    
    if selected_bucket != 'All':
        if selected_bucket == '0 – 250':
            filtered_df = filtered_df[filtered_df['c_final_price'] <= 250]
        elif selected_bucket == '250 – 500':
            filtered_df = filtered_df[(filtered_df['c_final_price'] > 250) & (filtered_df['c_final_price'] <= 500)]
        elif selected_bucket == '500 – 1,000':
            filtered_df = filtered_df[(filtered_df['c_final_price'] > 500) & (filtered_df['c_final_price'] <= 1000)]
        elif selected_bucket == '1,000 – 1,500':
            filtered_df = filtered_df[(filtered_df['c_final_price'] > 1000) & (filtered_df['c_final_price'] <= 1500)]
        elif selected_bucket == '1,500 – 3,000':
            filtered_df = filtered_df[(filtered_df['c_final_price'] > 1500) & (filtered_df['c_final_price'] <= 3000)]
        elif selected_bucket == '3,000+':
            filtered_df = filtered_df[filtered_df['c_final_price'] > 3000]
    
    if use_custom:
        filtered_df = filtered_df[
            (filtered_df['c_final_price'] >= min_price) & 
            (filtered_df['c_final_price'] <= max_price)
        ]
    
    if selected_sort == 'Max price diff (desc)':
        filtered_df = filtered_df.sort_values(
            ['price_diff', 'domain_priority', 'c_season_priority'], 
            ascending=[False, True, True],
            na_position='last'
        )
    elif selected_sort == 'Cettire price (desc)':
        filtered_df = filtered_df.sort_values('c_final_price', ascending=False, na_position='last')
    elif selected_sort == 'Cettire price (asc)':
        filtered_df = filtered_df.sort_values('c_final_price', ascending=True, na_position='last')
    elif selected_sort == 'Bally AU price (desc)':
        filtered_df = filtered_df.sort_values('m_final_price', ascending=False, na_position='last')
    elif selected_sort == 'Bally AU price (asc)':
        filtered_df = filtered_df.sort_values('m_final_price', ascending=True, na_position='last')
    elif selected_sort == 'Title A-Z':
        filtered_df = filtered_df.sort_values('c_title', na_position='last')
    elif selected_sort == 'Season priority':
        filtered_df = filtered_df.sort_values('c_season_priority', na_position='last')
    elif selected_sort == 'Site priority':
        filtered_df = filtered_df.sort_values('domain_priority', na_position='last')
    
    items_per_page = 50
    total_items = len(filtered_df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = 1
    
    st.markdown(f'<div class="result-count">Showing {min(items_per_page, total_items)} of {total_items} results</div>', unsafe_allow_html=True)
    
    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button('⏮ First', disabled=(st.session_state.current_page == 1)):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        if st.button('◀ Prev', disabled=(st.session_state.current_page == 1)):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col3:
        st.markdown(f'<div style="text-align: center; padding-top: 0.5rem;">Page {st.session_state.current_page} of {total_pages}</div>', unsafe_allow_html=True)
    
    with col4:
        if st.button('Next ▶', disabled=(st.session_state.current_page == total_pages)):
            st.session_state.current_page += 1
            st.rerun()
    
    with col5:
        if st.button('Last ⏭', disabled=(st.session_state.current_page == total_pages)):
            st.session_state.current_page = total_pages
            st.rerun()
    
    # Display products
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    for idx, row in page_df.iterrows():
        col_left, col_right = st.columns(2)
        with col_left:
            render_product_card(row, side='left')
        with col_right:
            render_product_card(row, side='right')
        st.markdown('<div class="tile-separator"></div>', unsafe_allow_html=True)


def analytics_page(df):
    """Render the Analytics page"""
    
    st.markdown('<h2 style="text-align: center; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 2rem;">ANALYTICS DASHBOARD</h2>', unsafe_allow_html=True)
    
    # Exclude OOS from numeric calculations
    df_analysis = df[~df['is_oos']].copy()
    df_price_valid = df_analysis[df_analysis['price_diff'].notna()].copy()
    
    # SITE-WISE COMPARISON
    st.markdown('<div class="section-header">Site-wise Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Average price difference (Cettire final price minus competitor price) and product count by competitor site. Negative values indicate Cettire is cheaper.</div>', unsafe_allow_html=True)
    
    if not df_price_valid.empty:
        site_stats = df_price_valid.groupby('domain_group').agg({
            'price_diff': ['mean', 'count'],
            'c_final_price': 'mean',
            'm_final_price': 'mean'
        }).round(2)
        
        site_stats.columns = ['Avg Price Diff (AUD)', 'Product Count', 'Avg Cettire Price', 'Avg Match Price']
        site_stats['% Cheaper on Cettire'] = ((df_price_valid.groupby('domain_group')['price_diff'].apply(lambda x: (x < 0).sum()) / 
                                                df_price_valid.groupby('domain_group').size()) * 100).round(1)
        
        site_stats = site_stats.sort_values('Avg Price Diff (AUD)', ascending=False)
        st.dataframe(site_stats, use_container_width=True)
        
        fig_site = px.bar(
            site_stats.reset_index(),
            x='domain_group',
            y='Avg Price Diff (AUD)',
            title='Average Price Difference by Site',
            labels={'domain_group': 'Site', 'Avg Price Diff (AUD)': 'Avg Price Diff (AUD)'},
            color='Avg Price Diff (AUD)',
            color_continuous_scale=['#C41E3A', '#FFFFFF', '#000000']
        )
        fig_site.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', size=12),
            showlegend=False
        )
        st.plotly_chart(fig_site, use_container_width=True)
    else:
        st.warning("No valid price comparison data available.")
    
    # CATEGORY-WISE ANALYSIS
    st.markdown('<div class="section-header">Category-wise Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Average price difference and product distribution across categories.</div>', unsafe_allow_html=True)
    
    if not df_price_valid.empty:
        category_stats = df_price_valid.groupby('category').agg({
            'price_diff': ['mean', 'count']
        }).round(2)
        category_stats.columns = ['Avg Price Diff (AUD)', 'Product Count']
        category_stats = category_stats.sort_values('Avg Price Diff (AUD)', ascending=False)
        
        st.dataframe(category_stats, use_container_width=True)
        
        fig_category = px.bar(
            category_stats.reset_index(),
            x='category',
            y='Avg Price Diff (AUD)',
            title='Average Price Difference by Category',
            labels={'category': 'Category', 'Avg Price Diff (AUD)': 'Avg Price Diff (AUD)'},
            color='Avg Price Diff (AUD)',
            color_continuous_scale=['#C41E3A', '#FFFFFF', '#000000']
        )
        fig_category.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter', size=12),
            showlegend=False
        )
        st.plotly_chart(fig_category, use_container_width=True)
    else:
        st.warning("No valid price comparison data available.")
    
    # SEASON TAG ANALYSIS
    st.markdown('<div class="section-header">Season Tag Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Price differences segmented by season tag on Cettire products.</div>', unsafe_allow_html=True)
    
    if not df_price_valid.empty:
        season_stats = df_price_valid.groupby('c_season_tag').agg({
            'price_diff': ['mean', 'count']
        }).round(2)
        season_stats.columns = ['Avg Price Diff (AUD)', 'Product Count']
        season_stats = season_stats.sort_values('Avg Price Diff (AUD)', ascending=False)
        
        st.dataframe(season_stats, use_container_width=True)
    else:
        st.warning("No valid price comparison data available.")
    
    # PRICE DIFFERENCE DISTRIBUTION
    st.markdown('<div class="section-header">Price Difference Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Distribution of price differences showing spread and outliers. Histogram shows frequency, boxplot shows quartiles and outliers.</div>', unsafe_allow_html=True)
    
    if not df_price_valid.empty:
        col1, col2 = st.columns(2)
        
        with col