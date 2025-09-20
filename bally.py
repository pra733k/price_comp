# scraper_bally.py (v0.1)
from typing import List, Dict, Optional
import requests, time, logging, re
from selectolax.parser import HTMLParser
from utils import ua, price_to_float, proxies_if_enabled
from config import BALLY_PRODUCT_URLS, SELECTORS, USE_ROTATING_PROXY, HTTP_PROXY_URL, HTTPS_PROXY_URL

def get_bally_products() -> List[Dict]:
    all_items = []
    seen = set()
    proxies = proxies_if_enabled(USE_ROTATING_PROXY, HTTP_PROXY_URL, HTTPS_PROXY_URL)

    for url in BALLY_PRODUCT_URLS:
        page_url = url
        while True:
            logging.info(f"[BALLY] Fetch: {page_url}")
            html = _get(page_url, proxies)
            if not html:
                break
            items, next_url = _parse_listing(html, page_url)
            new = 0
            for it in items:
                if it["url"] in seen:
                    continue
                seen.add(it["url"])
                all_items.append(it)
                new += 1
            if not next_url or new == 0:
                break
            page_url = next_url
            time.sleep(1.2)  # throttle
    return all_items

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

def _parse_listing(html: str, base_url: str):
    sel = SELECTORS.get("bally", {})
    tree = HTMLParser(html)
    items = []

    product_cards = tree.css(sel["product_card"]) if sel.get("product_card") else tree.css("a[href*='/en/']")
    for card in product_cards:
        try:
            href = card.attributes.get("href", "")
            if href and href.startswith("/"):
                href = "https://www.bally.com.au" + href
            title = None
            price = None
            img = None
            gender = "Unknown"

            if sel.get("title"):
                t = card.css_first(sel["title"])
                title = t.text().strip() if t else None
            if not title:
                title = card.text().strip()[:120]

            if sel.get("price"):
                p = card.css_first(sel["price"])
                price = price_to_float(p.text()) if p else None

            if sel.get("image"):
                ii = card.css_first(sel["image"])
                img = ii.attributes.get("src") if ii else None
            if not img:
                img_node = card.css_first("img")
                if img_node:
                    img = img_node.attributes.get("src") or img_node.attributes.get("data-src")

            # gender inference from URL path
            if "/men" in base_url.lower():
                gender = "Men"
            elif "/women" in base_url.lower():
                gender = "Women"

            items.append({
                "name": title or "Unknown",
                "url": href,
                "price": price,
                "description": None,
                "image_url": img,
                "product_code": None,
                "gender": gender
            })
        except Exception as e:
            pass

    # Find "next" pagination if present
    next_url = None
    if sel.get("next_button"):
        n = tree.css_first(sel["next_button"])
        if n and n.attributes.get("href"):
            next_url = n.attributes["href"]
            if next_url.startswith("/"):
                next_url = "https://www.bally.com.au" + next_url
    else:
        # heuristic: look for rel=next
        for a in tree.css("a[rel='next']"):
            href = a.attributes.get("href")
            if href:
                next_url = href if href.startswith("http") else "https://www.bally.com.au" + href
                break
    return items, next_url
