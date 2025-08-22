import os, re, json, time, csv, argparse
from pathlib import Path
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "OP-Scraper/1.1 (+personal research, CC-BY-SA compliance)"}
FANDOM_API = "https://onepiece.fandom.com/api.php"
FANDOM_PAGE_URL = "https://onepiece.fandom.com/wiki/Chapter_{id}"
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, data: Any): path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
def load_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))

def mediawiki_parse_page(chapter_id: int, throttle: float = 0.5) -> Dict[str, Any]:
    cache_path = CACHE_DIR / f"chapter_{chapter_id}.json"
    if cache_path.exists(): return load_json(cache_path)
    params = {
        "action": "parse",
        "page": f"Chapter_{chapter_id}",
        "prop": "text|sections|revid|displaytitle|wikitext",
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(FANDOM_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    save_json(cache_path, data)
    time.sleep(throttle)
    return data

def extract_infobox_fields(soup: BeautifulSoup) -> Dict[str, Any]:
    info = {}
    #portable infobox
    aside = soup.find("aside", class_=lambda c: c and "portable-infobox" in c)
    if aside:
        for dd in aside.select(".pi-data"):
            label_el = dd.select_one(".pi-data-label")
            value_el = dd.select_one(".pi-data-value")
            if not label_el or not value_el: continue
            label = label_el.get_text(" ", strip=True).lower()
            value = value_el.get_text(" ", strip=True)
            if ("title" in label) and ("japanese" not in label) and ("romaji" not in label):
                info["title_en"] = value
            elif "japanese" in label and "title" in label:
                info["title_jp"] = value
            elif "romaji" in label:
                info["title_romaji"] = value
            elif "pages" in label:
                m = re.search(r"\d+", value); 
                if m: info["pages"] = int(m.group(0))
            elif "release" in label or "published" in label:
                info["wsj_release_date_jp"] = value
            elif "volume" in label:
                info["volume"] = value
            elif "arc" in label:
                info["arc"] = value
            elif "issue" in label:
                info["wsj_issue"] = value
    #fallback: table infobox
    if not info:
        table = soup.find("table", class_=lambda c: c and "infobox" in c)
        if table:
            for tr in table.select("tr"):
                th, td = tr.find("th"), tr.find("td")
                if not th or not td: continue
                label = th.get_text(" ", strip=True).lower()
                value = td.get_text(" ", strip=True)
                if "title" in label and "japanese" not in label and "romaji" not in label:
                    info["title_en"] = value
                elif "japanese" in label and "title" in label:
                    info["title_jp"] = value
                elif "romaji" in label:
                    info["title_romaji"] = value
                elif "pages" in label:
                    m = re.search(r"\d+", value); 
                    if m: info["pages"] = int(m.group(0))
                elif "release date" in label or "published" in label:
                    info["wsj_release_date_jp"] = value
                elif "volume" in label:
                    info["volume"] = value
                elif "arc" in label:
                    info["arc"] = value
                elif "issue" in label:
                    info["wsj_issue"] = value
    return info

SUMMARY_HEADINGS = ["summary", "short summary", "long summary", "synopsis", "plot"]

def extract_section_html(soup: BeautifulSoup, titles: List[str]) -> str:
    for header_tag in soup.select("h2, h3"):
        span = header_tag.find("span", class_="mw-headline")
        if not span: continue
        headline = span.get_text(" ", strip=True).lower()
        if any(t == headline for t in titles):
            contents = []
            node = header_tag.next_sibling
            while node:
                if getattr(node, "name", None) in ("h2","h3"): break
                contents.append(str(node))
                node = node.next_sibling
            return "\n".join(contents).strip()
    return ""

def html_to_text(html_str: str) -> str:
    soup = BeautifulSoup(html_str, "lxml")
    for tag in soup.select("table, .infobox, .portable-infobox, .toc, .mw-editsection, .navbox"): tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def simple_paraphrase(text: str, max_sentences: int = 3, max_chars: int = 700) -> str:
    text = re.sub(r'“[^”]+”|"[^"]+"|\'[^\']+\'', "", text)
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    kept = []
    for s in sents:
        if len(s) < 15: continue
        kept.append(s)
        if len(kept) >= max_sentences: break
    out = " ".join(kept)
    if len(out) > max_chars: out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out

def extract_links_texts(html_str: str) -> List[str]:
    soup = BeautifulSoup(html_str or "", "lxml")
    names = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/"):
            t = a.get_text(" ", strip=True)
            if t and not re.fullmatch(r"\d+", t):
                names.append(t)
    seen, dedup = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); dedup.append(n)
    return dedup[:12]

def first_paragraph_fallback(soup: BeautifulSoup, max_paragraphs: int = 3) -> str:
    content = soup.select_one(".mw-parser-output")
    if not content: return ""
    paras = []
    for el in content.find_all(["p"]):
        txt = el.get_text(" ", strip=True)
        if len(txt) > 30:
            paras.append(txt)
        if len(paras) >= max_paragraphs: break
    return " ".join(paras)

def build_record(ch_id: int, html_text: str, parse_meta: Dict[str, Any]) -> Dict[str, Any]:
    soup = BeautifulSoup(html_text, "lxml")
    info = extract_infobox_fields(soup)
    summary_html = extract_section_html(soup, SUMMARY_HEADINGS)
    if not summary_html:
        fp = first_paragraph_fallback(soup)
        summary_text = fp
        links_src_html = str(soup.select_one(".mw-parser-output") or "")
    else:
        summary_text = html_to_text(summary_html)
        links_src_html = summary_html

    paraphrase = simple_paraphrase(summary_text) if summary_text else ""

    arc = info.get("arc", "")
    if not arc:
        catlinks = soup.select_one("#catlinks")
        if catlinks:
            cats = [a.get_text(" ", strip=True) for a in catlinks.select("a")]
            for c in cats:
                if c.endswith("Arc"):
                    arc = c; break

    names_from_links = extract_links_texts(links_src_html)

    title_en = info.get("title_en", "")
    if not title_en and "parse" in parse_meta and "displaytitle" in parse_meta["parse"]:
        dt = BeautifulSoup(parse_meta["parse"]["displaytitle"], "lxml").get_text(" ", strip=True)
        title_en = dt

    record = {
        "chapter_id": ch_id,
        "title_en": title_en if title_en else f"Chapter {ch_id}",
        "title_jp": info.get("title_jp", ""),
        "title_romaji": info.get("title_romaji", ""),
        "arc": arc,
        "volume": info.get("volume", ""),
        "viz_release_date": "",
        "wsj_release_date_jp": info.get("wsj_release_date_jp", ""),
        "wsj_issue": info.get("wsj_issue", ""),
        "pages": info.get("pages", ""),
        "summary_short_paraphrase": paraphrase,
        "beats_predicate": [],
        "characters_core": names_from_links,
        "locations_core": [],
        "threads_opened": [],
        "threads_closed": [],
        "source_url": FANDOM_PAGE_URL.format(id=ch_id),
        "source_attribution_title": f"Chapter {ch_id} (One Piece Wiki)",
        "source_attribution_author": "One Piece Wiki contributors",
        "source_license": "CC BY-SA"
    }
    return record

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows: return
    keys = list(rows[0].keys())
    import csv
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1157)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--throttle", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "onepiece_chapters.jsonl"
    existing_ids = set()
    if args.resume and jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try: existing_ids.add(int(json.loads(line).get("chapter_id", -1)))
                except: pass

    all_rows = []
    for ch_id in range(args.start, args.end + 1):
        if args.resume and ch_id in existing_ids: continue
        data = mediawiki_parse_page(ch_id, throttle=args.throttle)
        if "parse" not in data:
            row = {"chapter_id": ch_id, "title_en": f"Chapter {ch_id}", "title_jp":"", "title_romaji":"",
                   "arc":"", "volume":"", "viz_release_date":"", "wsj_release_date_jp":"", "wsj_issue":"", "pages":"",
                   "summary_short_paraphrase":"", "beats_predicate":[], "characters_core":[], "locations_core":[],
                   "threads_opened":[], "threads_closed":[], "source_url": FANDOM_PAGE_URL.format(id=ch_id),
                   "source_attribution_title": f"Chapter {ch_id} (One Piece Wiki)",
                   "source_attribution_author":"One Piece Wiki contributors", "source_license":"CC BY-SA",
                   "note": f"parse missing: {data.get('error','unknown')}"
                   }
        else:
            html_text = data["parse"]["text"]["*"]
            row = build_record(ch_id, html_text, data)
        all_rows.append(row)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(out_dir / "onepiece_chapters.csv", all_rows)
    (out_dir / "README.md").write_text(
        "Export created with OP Scraper v1.1. CC BY-SA sources; dataset must be CC BY-SA.\n", encoding="utf-8"
    )
    print("Done v1.1")

if __name__ == "__main__":
    main()
