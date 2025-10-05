# live_scrape.py
# BALLY AU PDP SCRAPER (requests first, robust DOM & JSON-LD parsing, optional Playwright fallback)
# - Reads matchlink333.csv and processes only rows where domain == www.bally.com.au and matchlink != NA
# - Extracts: m_title, m_retail_price, m_sale_price, m_image-src, m_season_tag, m_product_url, m_product_id
# - Correctly detects "Original" (line-through) vs "Discounted" (text-red) prices:
#     <span class="line-through">AUD$ 600</span>  -> retail
#     <span class="text-red"><span>AUD$ 360</span> ... </span> -> sale
# - Avoids picking the _next/image animation GIF; prefers product gallery / JSON-LD image URLs
# - LIMIT_ROWS lets you test quickly; set to None for full run.

from __future__ import annotations
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# ------------- CONFIG -------------
INPUT_CSV: str = "matchlink333.csv"
OUTPUT_CSV: str = "livescrape.csv"

DOMAIN_FILTER: str = "www.bally.com.au"
LIMIT_ROWS: Optional[int] = None   # <-- set to None to process ALL eligible rows

MAX_WORKERS: int = 6
REQ_TIMEOUT: int = 25
REQ_RETRIES: int = 3
REQ_BACKOFF: float = 0.8

PLAYWRIGHT_FALLBACK: bool = True   # Uses Playwright sequentially only for rows that still miss fields
PLAYWRIGHT_TIMEOUT_MS: int = 30000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
    "Cache-Control": "no-cache",
}

# ------------- HELPERS -------------
PRICE_NUM = re.compile(r"(\d+(?:[\.,]\d{1,2})?)")

def clean_price(text: str) -> Optional[float]:
    if not text:
        return None
    # keep first number (handles "AUD$ 360", "A$ 1,650.00", etc.)
    m = PRICE_NUM.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

def abs_url(url: str) -> str:
    if not url:
        return url
    # Reject _next/image proxy & animation gif
    if "/_next/image" in url or "animation" in url:
        return ""
    # Ensure protocol if missing (Bally returns absolute URLs normally)
    if url.startswith("//"):
        return "https:" + url
    return url

def pick_best_image(soup: BeautifulSoup, jsonld_first: Optional[str]) -> str:
    # 1) Prefer JSON-LD image if it looks like a product image
    if jsonld_first:
        url = abs_url(jsonld_first)
        if url and "animation" not in url and ".svg" not in url and "_next/image" not in url:
            return url

    # 2) Scan all <img> candidates and pick the most product-like
    candidates: List[Tuple[int, str]] = []
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        srcset = img.get("srcset") or ""
        for cand in filter(None, [src] + [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]):
            u = abs_url(cand)
            if not u:
                continue
            lower = u.lower()
            # heuristics: avoid logos, icons, placeholders, animation
            bad = any(x in lower for x in ["sprite", "logo", "icon", ".svg", "placeholder", "animation", "/_next/image"])
            if bad:
                continue
            # prefer bigger-looking images (very rough score by length)
            score = len(u)
            candidates.append((score, u))

    if candidates:
        candidates.sort(reverse=True)  # longest first
        return candidates[0][1]
    return ""

def extract_from_jsonld(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    data: Dict[str, Optional[str]] = {"title": None, "image": None, "price": None, "currency": None}
    for s in soup.select('script[type="application/ld+json"]'):
        try:
            obj = json.loads(s.string or "")
        except Exception:
            continue
        # sometimes array
        objs = obj if isinstance(obj, list) else [obj]
        for o in objs:
            if not isinstance(o, dict):
                continue
            if o.get("@type") == "Product":
                name = o.get("name")
                image = o.get("image")
                if isinstance(image, list) and image:
                    image = image[0]
                offers = o.get("offers") or {}
                if isinstance(offers, list) and offers:
                    offers = offers[0]
                price = offers.get("price")
                currency = offers.get("priceCurrency")
                data["title"] = data["title"] or name
                data["image"] = data["image"] or image
                data["price"] = data["price"] or (str(price) if price else None)
                data["currency"] = data["currency"] or currency
    return data

def extract_prices_dom(soup: BeautifulSoup) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (retail_price, sale_price).
    Handles the Bally AU DOM you showed:
      <span class="line-through">AUD$ 600</span>  (original)
      <span class="text-red"><span>AUD$ 360</span> ... </span>  (discounted)
    """
    retail = None
    sale = None

    # A) explicit line-through + text-red block
    # look around any container that includes "Original Price" sr-only text too
    price_blocks = soup.find_all(lambda tag:
                                 tag.name in ("article", "div", "section") and
                                 ("line-through" in " ".join(tag.get("class", [])) or
                                  tag.find(class_=re.compile(r"line-through")) or
                                  tag.find(class_=re.compile(r"text-red")) or
                                  (tag.find("span", string=re.compile("Original Price", re.I)) is not None)))
    # include whole document as fallback scan target if len==0
    if not price_blocks:
        price_blocks = [soup]

    for block in price_blocks:
        # original (retail) price
        line = block.find("span", class_=re.compile(r"line-through"))
        if line and retail is None:
            retail = clean_price(line.get_text(" ", strip=True))

        # discounted (sale) price lives inside text-red span (inner span holds number)
        red = block.find("span", class_=re.compile(r"text-red"))
        if red and sale is None:
            # primary number inside the red block
            sale = clean_price(red.get_text(" ", strip=True))

        # support sr-only labels explicitly
        if retail is None:
            sr = block.find("span", string=re.compile("Original Price", re.I))
            if sr:
                next_num = sr.find_next("span")
                retail = clean_price(next_num.get_text(" ", strip=True)) if next_num else None
        if sale is None:
            sr2 = block.find("span", string=re.compile("Discounted Price", re.I))
            if sr2:
                next_num2 = sr2.find_next("span")
                sale = clean_price(next_num2.get_text(" ", strip=True)) if next_num2 else None

        if retail or sale:
            break

    # B) If only one price visible (not on sale): set retail=visible, sale=None
    if retail is None and sale is not None:
        # we found a red price but no line-through -> it's actually the current price;
        # treat it as retail (no sale)
        retail, sale = sale, None
    elif retail is None and sale is None:
        # try any prominent price number on page as retail
        any_price = None
        for cand in soup.find_all(text=PRICE_NUM):
            val = clean_price(str(cand))
            if val:
                any_price = val
                break
        retail = any_price

    return retail, sale

def extract_title(soup: BeautifulSoup, jsonld_title: Optional[str]) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        if t:
            return t
    return jsonld_title

def extract_season(soup: BeautifulSoup) -> str:
    # If Bally exposes season in the attributes table/specs, capture it.
    # Otherwise return empty string.
    text = soup.get_text(" ", strip=True)
    m = re.search(r"(Spring|Summer|Fall|Autumn|Winter)\s*(\d{2,4})", text, flags=re.I)
    return m.group(0) if m else ""

def product_id_from_url(url: str) -> str:
    m = re.search(r"(\d{6,})", url)
    return m.group(1) if m else ""

def fetch_html(url: str) -> Optional[str]:
    for attempt in range(1, REQ_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.text
        except Exception:
            if attempt == REQ_RETRIES:
                return None
            time.sleep(REQ_BACKOFF * attempt)
    return None

# ------------- PLAYWRIGHT FALLBACK -------------
def fetch_with_playwright(urls: List[str]) -> Dict[str, str]:
    """Fetch a small list of URLs sequentially via Playwright to get fully rendered HTML."""
    out: Dict[str, str] = {}
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return out

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=HEADERS["User-Agent"],
                                      viewport={"width": 1280, "height": 900})
        page = context.new_page()
        for url in urls:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
                # give price widgets a moment
                page.wait_for_timeout(800)
                out[url] = page.content()
            except Exception:
                out[url] = ""
        context.close()
        browser.close()
    return out

# ------------- PIPELINE -------------
@dataclass
class RowOut:
    m_title: str = ""
    m_retail_price: Optional[float] = None
    m_sale_price: Optional[float] = None
    m_image_src: str = ""
    m_season_tag: str = ""
    m_product_url: str = ""
    m_product_id: str = ""

def parse_bally_pdp(url: str, html: str) -> RowOut:
    soup = BeautifulSoup(html, "html.parser")

    # JSON-LD helpers
    j = extract_from_jsonld(soup)
    jsonld_title = j.get("title")
    jsonld_image = j.get("image")
    # (jsonld price is the current price; we prefer DOM to split retail/sale clearly)

    # Title
    title = extract_title(soup, jsonld_title) or ""

    # Prices (robust DOM parsing according to provided HTML)
    retail, sale = extract_prices_dom(soup)

    # Image (avoid _next/image / animation)
    image = pick_best_image(soup, jsonld_image)

    # Season (best-effort, often empty)
    season = extract_season(soup)

    return RowOut(
        m_title=title,
        m_retail_price=retail,
        m_sale_price=sale,
        m_image_src=image,
        m_season_tag=season,
        m_product_url=url,
        m_product_id=product_id_from_url(url),
    )

def read_bally_rows(csv_path: str, limit: Optional[int]) -> List[str]:
    urls: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            dom = (row.get("domain") or "").strip()
            link = (row.get("matchlink") or "").strip()
            if dom == DOMAIN_FILTER and link and link.upper() != "NA":
                urls.append(link)
                if limit is not None and len(urls) >= limit:
                    break
    return urls

def main():
    urls = read_bally_rows(INPUT_CSV, LIMIT_ROWS)

    print("=" * 80)
    print("BALLY AU PDP SCRAPER (requests + JSON-LD + Next.js + Playwright fallback)")
    print("=" * 80)
    print(f"Loaded {len(urls)} rows from {INPUT_CSV}")
    print(f"DOMAIN         : {DOMAIN_FILTER}")
    print(f"LIMIT_ROWS     : {'FULL' if LIMIT_ROWS is None else LIMIT_ROWS}")
    print(f"CONCURRENCY    : {MAX_WORKERS}")
    print(f"PLAYWRIGHT FB? : {PLAYWRIGHT_FALLBACK}")
    print(f"Output         : {OUTPUT_CSV}\n")

    t0 = time.time()
    out_rows: Dict[str, RowOut] = {}
    errors: Counter = Counter()
    needs_fb: List[str] = []

    # Pass 1: requests (fast)
    def worker(u: str) -> Tuple[str, Optional[RowOut], Optional[str]]:
        html = fetch_html(u)
        if not html:
            return u, None, "requests_failed"
        try:
            row = parse_bally_pdp(u, html)
        except Exception:
            return u, None, "parse_error"
        # Check missing criticals -> fallback later
        if not row.m_title or (row.m_retail_price is None and row.m_sale_price is None) or not row.m_image_src:
            return u, row, "needs_fallback"
        return u, row, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, u): u for u in urls}
        done = 0
        for fut in as_completed(futures):
            u, row, err = fut.result()
            done += 1
            print(f"[{done}/{len(urls)}] {'✓' if not err else '•'} {u}")
            if err == "requests_failed":
                errors[err] += 1
                needs_fb.append(u)
            elif err == "parse_error":
                errors[err] += 1
                needs_fb.append(u)
            elif err == "needs_fallback":
                # keep the partial row to merge later after FB
                out_rows[u] = row
                needs_fb.append(u)
            else:
                out_rows[u] = row

    # Pass 2: Playwright fallback for rows still missing critical fields
    if PLAYWRIGHT_FALLBACK and needs_fb:
        fb_html = fetch_with_playwright(needs_fb)
        for u in needs_fb:
            html = fb_html.get(u, "")
            if not html:
                continue
            try:
                row_fb = parse_bally_pdp(u, html)
                # Merge: prefer FB for missing fields
                cur = out_rows.get(u, RowOut(m_product_url=u, m_product_id=product_id_from_url(u)))
                if not cur.m_title and row_fb.m_title:
                    cur.m_title = row_fb.m_title
                if cur.m_retail_price is None and row_fb.m_retail_price is not None:
                    cur.m_retail_price = row_fb.m_retail_price
                if cur.m_sale_price is None and row_fb.m_sale_price is not None:
                    cur.m_sale_price = row_fb.m_sale_price
                if not cur.m_image_src and row_fb.m_image_src:
                    cur.m_image_src = row_fb.m_image_src
                if not cur.m_season_tag and row_fb.m_season_tag:
                    cur.m_season_tag = row_fb.m_season_tag
                if not cur.m_product_id and row_fb.m_product_id:
                    cur.m_product_id = row_fb.m_product_id
                out_rows[u] = cur
            except Exception:
                errors["fallback_parse_error"] += 1

    # Post-fix: if page not on sale, ensure retail holds the visible price & sale stays blank
    for u, r in out_rows.items():
        if r.m_retail_price is None and r.m_sale_price is not None:
            r.m_retail_price, r.m_sale_price = r.m_sale_price, None

    # Write CSV (keep URL order)
    cols = [
        "m_title",
        "m_retail_price",
        "m_sale_price",
        "m_image-src",
        "m_season_tag",
        "m_product_url",
        "m_product_id",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for u in urls:
            r = out_rows.get(u, RowOut(m_product_url=u, m_product_id=product_id_from_url(u)))
            w.writerow([
                r.m_title,
                f"{r.m_retail_price:.2f}" if isinstance(r.m_retail_price, (int, float)) else "",
                f"{r.m_sale_price:.2f}" if isinstance(r.m_sale_price, (int, float)) else "",
                r.m_image_src,
                r.m_season_tag,
                r.m_product_url,
                r.m_product_id,
            ])

    dt = time.time() - t0
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Rows written : {len(urls)}")
    ok_fast = sum(1 for r in out_rows.values() if r.m_title and (r.m_retail_price is not None or r.m_sale_price is not None))
    print(f"OK (fast)    : {ok_fast}")
    print(f"Failed (fast): {len(urls) - ok_fast}")
    print(f"Time taken   : {int(dt//60)}m {int(dt%60)}s")
    print(f"CSV          : {OUTPUT_CSV}\n")

    # Error pivot
    if errors:
        print("ERROR PIVOT:")
        for k, v in errors.items():
            print(f"  {k:>22} : {v}")
    else:
        print("ERROR PIVOT: none 🎉")


if __name__ == "__main__":
    main()
