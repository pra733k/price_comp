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
