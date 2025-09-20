# scraper_cettire.py (v0.1)
from typing import List, Dict, Optional
import requests, time, logging
from selectolax.parser import HTMLParser
from bs4 import BeautifulSoup  # optional
from utils import ua, price_to_float, proxies_if_enabled
from config import SELECTORS, CETTIRE_LISTING_URL, USE_ROTATING_PROXY, HTTP_PROXY_URL, HTTPS_PROXY_URL

def get_cettire_products() -> List[Dict]:
    """
    Returns a list of dicts with keys:
    name, url, price, description, image_url, product_code (if found), gender (Unknown)
    """
    out = []
    seen = set()
    page = 1
    base_url = CETTIRE_LISTING_URL
    proxies = proxies_if_enabled(USE_ROTATING_PROXY, HTTP_PROXY_URL, HTTPS_PROXY_URL)

    while True:
        url = _ensure_page(base_url, page)
        logging.info(f"[CETTIRE] Fetch page {page}: {url}")
        html = _get(url, proxies)
        if not html:
            break
        items = _parse_listing(html)
        new = 0
        for it in items:
            if it["url"] in seen:
                continue
            seen.add(it["url"])
            out.append(it)
            new += 1
        if new == 0:
            break
        page += 1
        time.sleep(1.2)  # throttle
    return out

def _ensure_page(url: str, page: int) -> str:
    # if there is a "page=" param, replace; else append
    if "page=" in url:
        import re
        return re.sub(r"page=\d+", f"page={page}", url)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}page={page}"

def _get(url: str, proxies):
    try:
        r = requests.get(url, timeout=25, proxies=proxies, headers={"User-Agent": ua()})
        if r.status_code != 200:
            logging.warning(f"HTTP {r.status_code} for {url}")
            return None
        return r.text
    except Exception as e:
        logging.error(f"GET failed {url}: {e}")
        return None

def _parse_listing(html: str) -> List[Dict]:
    sel = SELECTORS.get("cettire", {})
    tree = HTMLParser(html)
    items = []

    # Heuristic parsing; replace with locked selectors if you have them
    product_cards = tree.css(sel["product_card"]) if sel.get("product_card") else tree.css("a[href*='/products/']")
    for card in product_cards:
        try:
            href = card.attributes.get("href", "")
            if href and href.startswith("/"):
                href = "https://www.cettire.com" + href
            title = None
            price = None
            img = None

            # Title
            if sel.get("title"):
                t = card.css_first(sel["title"])
                title = t.text().strip() if t else None
            if not title:
                title = card.text().strip()[:120]

            # Price
            if sel.get("price"):
                p = card.css_first(sel["price"])
                price = price_to_float(p.text()) if p else None

            # Image heuristic
            if sel.get("image"):
                ii = card.css_first(sel["image"])
                img = ii.attributes.get("src") if ii else None
            if not img:
                img_node = card.css_first("img")
                if img_node:
                    img = img_node.attributes.get("src") or img_node.attributes.get("data-src")

            items.append({
                "name": title or "Unknown",
                "url": href,
                "price": price,
                "description": None,  # detail fetch optional (can slow down)
                "image_url": img,
                "product_code": None,
                "gender": "Unknown"
            })
        except Exception as e:
            logging.warning(f"Parse error: {e}")
    return items
