import os
import re
import time
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta, timezone
import feedparser
import httpx
import numpy as np
from opencc import OpenCC
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from google import genai

# 讀取環境變數
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()

DB_PATH = "news.db"
KEYWORDS_PATH = "keywords.json"
WIDGET_JSON_PATH = "widget_data.json"

# 簡轉繁與多語系專用翻譯工具 (日文強制指定 source='ja' 避免漢字誤判為中文)
cc = OpenCC('s2twp')
en_translator = GoogleTranslator(source='auto', target='zh-TW')
ja_translator = GoogleTranslator(source='ja', target='zh-TW')

# ==========================================
# 🗑️ 無用標題過濾表單 (同時過濾國內與國際新聞)
# ==========================================
JUNK_PATTERNS = [
    # 韓聯社與外媒常見無用標題/圖輯/純快訊空殼
    r"^\[?韓聯社\s*(快訊|今日要聞|要聞|簡訊|圖片|照片|圖輯|視頻|影音)\]?$",
    r"^韓聯社\s*[:：]?\s*(快訊|簡訊|圖片新聞|今日要聞)$",
    r"^\[(圖|圖片|照片|組圖|視頻|影音|專訪|特寫|圖表)\]$",
    r"韓聯社今日要聞",
    r"韓聯社新聞列表",
    r"韓國歷史上的",
    r"韓國主要報紙頭條新聞",
    r"^(快訊|詳訊|簡訊)\s*[:：]?\s*$",
    r"Yonhap\s+News\s+Summary",
    r"Top\s+News\s+Headlines",

    # Yahoo Finance 個股報價與歷史資訊頁面
    r"Stock Price,\s*News,\s*Quote\s*(&|and)\s*History",
    r"Stock Price.*Quote.*History",
    r"TITLE_QUOTE",

    # 紐約時報中文網、日經亞洲等外媒常見空標題或首頁標語
    r"^[-－—|\s]*(紐約時報|纽约时报|日經|日经|Nikkei|NHK|BBC|CNN|法廣|RFI|德國之聲|DW|韓聯社|FT|彭博|Bloomberg|路透|華爾街日報|WSJ).*?(中文網|中文网|中文|新聞|新聞網|Asia|World)?[-－—|\s]*$",
    r"Business,\s*politics,\s*economy\s*and\s*tech\s*news",
    r"商業、政治、經濟(和|與)科技新聞(與|和)分析",
    r"^News,\s*analysis\s*and\s*opinion\s*from",

    # 國內外媒體常見無意義綜合欄位/早晚報/影音導覽
    r"^(今日|每日|晨間|晚間|一週)?\s*(新聞總覽|即時新聞總覽|新聞提要|要聞總覽|重點新聞|熱門新聞|焦點新聞|早報|晚報|新聞快報)$",
    r"^(Morning|Evening|Daily)\s+(Briefing|Digest|Wrap|Roundup|Newsletter)",
    r"^\[?(直播|LIVE|Podcast|圖輯|影)\s*[:：\]]?$",
    r"^(工商時報|經濟日報|聯合報|自由時報|中時)\s*(社論|黑白集|短評)$"
    r"^(今日|本週)運勢",
    r"^(中獎|開獎)號碼",
]

# 台灣主流、科技、財經與重大訊息媒體 RSS
TW_FEEDS = {
    # --- 綜合主流頭條 ---
    "中央社": "https://feeds.feedburner.com/cnaFirstNews",
    "公視新聞": "https://news.google.com/rss/search?q=site:news.pts.org.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "聯合新聞": "https://udn.com/rssfeed/news/2/6638?ch=news",
    "自由時報": "https://news.ltn.com.tw/rss/all.xml",
    "中時新聞": "https://news.google.com/rss/search?q=site:chinatimes.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "ETtoday": "https://feeds.feedburner.com/ettoday/realtime",
    "關鍵評論網": "https://news.google.com/rss/search?q=site:thenewslens.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "轉角國際": "https://news.google.com/rss/search?q=site:global.udn.com+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "風傳媒": "https://news.google.com/rss/search?q=site:storm.mg+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "TVBS新聞": "https://news.google.com/rss/search?q=site:news.tvbs.com.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    # --- 台灣科技與半導體媒體 ---
    "科技新報": "https://technews.tw/feed/",
    "財經新報": "https://finance.technews.tw/feed/",
    "數位時代": "https://news.google.com/rss/search?q=site:bnext.com.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "iThome": "https://www.ithome.com.tw/rss",
    "INSIDE硬塞": "https://www.inside.com.tw/feed/rss",
    "科技報橘": "https://buzzorange.com/techorange/feed/",
    "癮科技Cool3c": "https://www.cool3c.com/rss",
    # --- 台灣財經、產業與上市櫃重大訊息 ---
    "公開資訊觀測站重訊": "https://news.google.com/rss/search?q=%22公開資訊觀測站%22+OR+%22重大訊息%22+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "經濟日報": "https://money.udn.com/rssfeed/news/1001/5588",
    "工商時報": "https://news.google.com/rss/search?q=site:ctee.com.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "鉅亨網": "https://news.google.com/rss/search?q=site:news.cnyes.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "MoneyDJ理財網": "https://news.google.com/rss/search?q=site:moneydj.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "天下雜誌": "https://www.cw.com.tw/RSS/cw_content.xml",
    "商業周刊": "https://news.google.com/rss/search?q=site:businessweekly.com.tw+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "今周刊": "https://news.google.com/rss/search?q=site:businesstoday.com.tw+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "財訊": "https://news.google.com/rss/search?q=site:wealth.com.tw+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
}

# 國際主流與科技財經媒體 RSS
INTL_FEEDS = {
    # --- 國際綜合主流 ---
    "BBC中文": "https://feeds.bbci.co.uk/zhongwen/trad/rss.xml",
    "紐約時報中文": "https://news.google.com/rss/search?q=site:cn.nytimes.com+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "德國之聲": "https://rss.dw.com/rdf/rss-chi-all",
    "法廣RFI": "https://www.rfi.fr/tw/rss",
    "NHK新聞": "https://www.nhk.or.jp/rss/news/cat0.xml",
    "韓聯社": "https://cn.yna.co.kr/RSS/news.xml",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "CNN World": "http://rss.cnn.com/rss/edition_world.rss",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "Washington Post": "https://news.google.com/rss/search?q=site:washingtonpost.com+when:1d&hl=en-US&gl=US&ceid=US:en",
    # --- 國際財經與美股權威 ---
    "FT中文網": "https://www.ftchinese.com/rss/feed",
    "Financial Times": "https://www.ft.com/world?format=rss",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "Bloomberg Tech": "https://feeds.bloomberg.com/technology/news.rss",
    "WSJ Markets": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories",
    "Yahoo Finance": "https://news.google.com/rss/search?q=site:finance.yahoo.com+when:1d&hl=en-US&gl=US&ceid=US:en",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
    "Nikkei Asia": "https://news.google.com/rss/search?q=site:asia.nikkei.com+when:1d&hl=en-US&gl=US&ceid=US:en",
    # --- 國際科技、AI 與半導體硬體 ---
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Wired": "https://www.wired.com/feed/rss",
    "VentureBeat": "https://news.google.com/rss/search?q=site:venturebeat.com+when:1d&hl=en-US&gl=US&ceid=US:en",
    "Tom's Hardware": "https://www.tomshardware.com/feeds/all",
    "Wccftech": "https://wccftech.com/feed/"
}

ALL_SOURCE_NAMES = set(TW_FEEDS.keys()) | set(INTL_FEEDS.keys()) | {
    "紐約時報中文網", "紐約時報", "日經亞洲", "日經中文網", "法國國際廣播電台"
}

def send_tg_message(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        httpx.post(url, json={
            "chat_id": TG_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=10.0)
    except Exception as e:
        print(f"Telegram 推播失敗: {e}")

def has_japanese(text):
    """偵測是否包含日文平假名 (\u3040-\u309f)、片假名 (\u30a0-\u30ff) 或半形假名 (\uff66-\uff9f)"""
    if not text:
        return False
    return any(('\u3040' <= ch <= '\u30ff') or ('\uff66' <= ch <= '\uff9f') for ch in text)

def is_mostly_english(text):
    """偵測是否包含日文假名或以英文/外文為主"""
    if not text:
        return False
    if has_japanese(text):
        return True
    zh_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    en_chars = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    return zh_chars == 0 or (en_chars > 12 and zh_chars < 4)

def ensure_traditional_chinese(text):
    text = (text or "").strip()
    if not text:
        return text
    # 1. 若含日文假名，強制使用 source='ja' 翻譯，徹底解決日文漢字被 auto 誤判為中文的問題
    if has_japanese(text):
        try:
            translated = ja_translator.translate(text)
            if translated:
                text = translated
        except Exception as e:
            print(f"日文翻譯備援失敗: {e}")
    # 2. 若為英文或其他外文，使用 auto 翻譯
    elif is_mostly_english(text):
        try:
            translated = en_translator.translate(text)
            if translated:
                text = translated
        except Exception as e:
            print(f"英文翻譯備援失敗: {e}")
    return cc.convert(text)

def strip_media_suffixes(title, is_google_news=False):
    """徹底剝除 Google News 雙重破折號尾綴 (如 '- 紐約時報中文網 - 紐約時報中文網') 與開頭破折號"""
    t = (title or "").strip()
    if is_google_news:
        while " - " in t:
            left, right = t.rsplit(" - ", 1)
            # 若左半邊為空，或右半邊為媒體名稱 (長度較短)，則持續剝除
            if not left.strip() or len(right.strip()) <= 25:
                t = left.strip()
            else:
                break
    # 移除開頭與結尾殘留的破折號或直線符號
    t = re.sub(r"^[-－—|｜:：\s]+", "", t)
    t = re.sub(r"[-－—|｜\s]+$", "", t)
    return t.strip()

def is_junk_title(raw_title, zh_title="", source=""):
    """
    檢查國內外新聞標題是否屬於無用/空白/純媒體名/純圖輯/網站標語標題
    """
    for text in (raw_title, zh_title):
        if not text:
            continue
        t = text.strip()
        # 去除頭尾破折號與括號標籤後檢查
        cleaned = strip_media_suffixes(t, is_google_news=True)
        cleaned = re.sub(r"^(\[[^\]]+\]|【[^】]+】|\([^\)]+\)|（[^）]+）)\s*", "", cleaned).strip()

        # 若清除後不足 6 個字，或剛好等於媒體名稱，直接判定為無效標題
        if len(cleaned) < 6:
            return True
        if cleaned in ALL_SOURCE_NAMES or (source and cleaned == source):
            return True

        for pattern in JUNK_PATTERNS:
            if re.search(pattern, t, re.IGNORECASE) or re.search(pattern, cleaned, re.IGNORECASE):
                return True
    return False

def clean_rss_title(raw_title, is_google_news=False):
    title = strip_media_suffixes(raw_title, is_google_news=is_google_news)
    return ensure_traditional_chinese(title)

# --- 防誤分群第二道防線：字面關鍵詞交集檢查 ---
STOPWORDS_CHARS = set("的了在是與及或將為對從向被把讓給比跟和但而卻又也還更最很太都只就才已曾正要會能可好大中小上下前後裡外內高低長短新舊多少全各每某該此其之以為於與及等後前月日年時分秒週今明昨")

def extract_key_tokens(title):
    zh_chunks = re.findall(r'[\u4e00-\u9fff]+', title)
    bigrams = set()
    chars = set()
    for chunk in zh_chunks:
        for ch in chunk:
            if ch not in STOPWORDS_CHARS:
                chars.add(ch)
        for i in range(len(chunk) - 1):
            bg = chunk[i:i+2]
            if not (bg[0] in STOPWORDS_CHARS and bg[1] in STOPWORDS_CHARS):
                bigrams.add(bg)
    en_tokens = set(re.findall(r'[a-zA-Z]{2,}[0-9]*', title.lower()))
    return bigrams, en_tokens, chars

def is_truly_related(title1, title2, cos_dist):
    if cos_dist < 0.12:
        return True
    bg1, en1, ch1 = extract_key_tokens(title1)
    bg2, en2, ch2 = extract_key_tokens(title2)
    if len(bg1 & bg2) >= 1:
        return True
    if len(en1 & en2) >= 1:
        return True
    if len(ch1 & ch2) >= 2 and cos_dist < 0.18:
        return True
    return False

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles (
                    link TEXT PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    category TEXT,
                    fetched_at TIMESTAMP
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS summary_cache (
                    cluster_id TEXT PRIMARY KEY,
                    article_links TEXT,
                    headline TEXT,
                    bullets TEXT,
                    translated_links TEXT,
                    created_at TIMESTAMP
                )''')
    try:
        c.execute("ALTER TABLE summary_cache ADD COLUMN translated_links TEXT")
    except sqlite3.OperationalError:
        pass

    # 1. 清洗歷史 articles：刪除空殼/無用標題，並強制將殘留日文或英文標題翻譯為繁中
    c.execute("SELECT link, title, source FROM articles")
    for link, title, source in c.fetchall():
        cleaned_t = strip_media_suffixes(title, is_google_news=True)
        if is_junk_title(title, cleaned_t, source):
            c.execute("DELETE FROM articles WHERE link = ?", (link,))
        elif is_mostly_english(cleaned_t) or cleaned_t != title:
            zh_title = ensure_traditional_chinese(cleaned_t)
            if is_junk_title(cleaned_t, zh_title, source) or is_mostly_english(zh_title):
                c.execute("DELETE FROM articles WHERE link = ?", (link,))
            else:
                c.execute("UPDATE articles SET title = ? WHERE link = ?", (zh_title, link))

    # 2. 清洗歷史 summary_cache：若快取標題或摘要含日文/外文、無用標題、或舊備援格式，一律刪除重算
    c.execute("SELECT cluster_id, headline, bullets FROM summary_cache")
    for cid, headline, bullets_json in c.fetchall():
        bullets_str = bullets_json or ""
        if (
            is_mostly_english(headline)
            or has_japanese(bullets_str)
            or is_junk_title(headline)
            or headline.startswith("-")
        ):
            c.execute("DELETE FROM summary_cache WHERE cluster_id = ?", (cid,))

    conn.commit()
    return conn

def fetch_mops_announcements(conn, now_iso):
    """透過證交所 OpenAPI 抓取最新「公開資訊觀測站（MOPS）」上市公司當日重大訊息"""
    c = conn.cursor()
    mops_articles = []
    api_url = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
    try:
        resp = httpx.get(api_url, timeout=15.0, follow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            for item in data[-25:]:
                co_id = (item.get("公司代號") or "").strip()
                co_name = (item.get("公司名稱") or "").strip()
                subject = (item.get("主旨") or "").strip().replace("\r", "").replace("\n", " ")
                date_str = (item.get("發言日期") or "").strip()
                time_str = (item.get("發言時間") or "").strip()
                if not co_id or not subject:
                    continue

                title = cc.convert(f"【重訊】{co_id} {co_name}：{subject}")
                if is_junk_title(title, source="公開資訊觀測站"):
                    continue

                uid = hashlib.md5(f"{co_id}_{date_str}_{time_str}_{subject[:20]}".encode("utf-8")).hexdigest()[:10]
                link = f"https://mops.twse.com.tw/mops/web/t05sr01_1#mops-{co_id}-{uid}"

                c.execute("SELECT 1 FROM articles WHERE link = ?", (link,))
                if not c.fetchone():
                    c.execute(
                        "INSERT INTO articles (link, title, source, category, fetched_at) VALUES (?, ?, ?, ?, ?)",
                        (link, title, "公開資訊觀測站", "tw", now_iso)
                    )
                    mops_articles.append({
                        "link": link,
                        "title": title,
                        "raw_title": title,
                        "source": "公開資訊觀測站",
                        "category": "tw"
                    })
    except Exception as e:
        print(f"MOPS OpenAPI 讀取略過: {e}")
    return mops_articles

def fetch_all_feeds(conn):
    c = conn.cursor()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    new_articles = []
    error_logs = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*"
    }

    for category, feed_dict in [("tw", TW_FEEDS), ("intl", INTL_FEEDS)]:
        for source, url in feed_dict.items():
            try:
                is_gn = "news.google.com" in url
                resp = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
                if resp.status_code >= 400:
                    error_logs.append(f"• <b>{source}</b>: HTTP {resp.status_code}")
                    continue
                feed = feedparser.parse(resp.content)
                if not feed.entries:
                    error_logs.append(f"• <b>{source}</b>: 解析無文章")
                    continue

                for entry in feed.entries[:15]:
                    link = entry.get("link", "").strip()
                    raw_title = entry.get("title", "").strip()
                    if not link or not raw_title:
                        continue

                    # 第一關：剝除尾綴後先檢查是否為空殼標題或無用標題 (如 "- 紐約時報中文網")
                    stripped_raw = strip_media_suffixes(raw_title, is_google_news=is_gn)
                    if is_junk_title(raw_title, stripped_raw, source):
                        continue

                    c.execute("SELECT 1 FROM articles WHERE link = ?", (link,))
                    if not c.fetchone():
                        title = clean_rss_title(raw_title, is_google_news=is_gn)
                        # 第二關：翻譯成繁中後再次檢查是否為網站口號或殘留未翻譯假名
                        if is_junk_title(stripped_raw, title, source):
                            continue

                        c.execute(
                            "INSERT INTO articles (link, title, source, category, fetched_at) VALUES (?, ?, ?, ?, ?)",
                            (link, title, source, category, now_iso)
                        )
                        new_articles.append({
                            "link": link,
                            "title": title,
                            "raw_title": stripped_raw,
                            "source": source,
                            "category": category
                        })
            except Exception as e:
                error_logs.append(f"• <b>{source}</b>: {type(e).__name__}")

    mops_new = fetch_mops_announcements(conn, now_iso)
    new_articles.extend(mops_new)

    conn.commit()
    return new_articles, error_logs

def check_keywords(new_articles):
    if not os.path.exists(KEYWORDS_PATH):
        return
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        keywords = json.load(f)
    if not keywords:
        return

    for a in new_articles:
        raw_t = a.get("raw_title", a["title"]).lower()
        zh_t = a["title"].lower()
        matched = [kw for kw in keywords if kw.lower() in raw_t or kw.lower() in zh_t]
        if matched:
            msg = (
                f"🚨 <b>關鍵字命中：{', '.join(matched)}</b>\n\n"
                f"📰 {a['title']}\n"
                f"🏢 來源：{a['source']}\n"
                f"🔗 <a href='{a['link']}'>點此閱讀報導</a>"
            )
            send_tg_message(msg)

def find_cached_summary(conn, current_links):
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    c.execute("SELECT article_links, headline, bullets, translated_links FROM summary_cache WHERE created_at >= ?", (cutoff,))
    rows = c.fetchall()

    curr_set = set(current_links)
    for row_links_json, headline, bullets_json, trans_links_json in rows:
        if is_mostly_english(headline) or is_junk_title(headline):
            continue
        bullets_list = json.loads(bullets_json) if bullets_json else []
        # 若快取中的摘要含有日文假名或只是「媒體：標題」的失敗備援，不使用該快取
        if any(has_japanese(b) for b in bullets_list):
            continue
        if bullets_list and any(bullets_list[0].startswith(f"{src}：") for src in ALL_SOURCE_NAMES):
            continue

        cached_set = set(json.loads(row_links_json))
        intersection = len(curr_set & cached_set)
        if curr_set == cached_set or (intersection >= 2 and intersection / max(len(curr_set), 1) >= 0.6):
            return {
                "headline": ensure_traditional_chinese(headline),
                "bullets": [ensure_traditional_chinese(b) for b in bullets_list],
                "translated_links": json.loads(trans_links_json) if trans_links_json else None
            }
    return None

def save_summary_cache(conn, current_links, headline, bullets, translated_links):
    c = conn.cursor()
    sorted_links = sorted(list(set(current_links)))
    cluster_id = hashlib.md5("".join(sorted_links).encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        "INSERT OR REPLACE INTO summary_cache (cluster_id, article_links, headline, bullets, translated_links, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            cluster_id,
            json.dumps(sorted_links),
            cc.convert(headline),
            json.dumps([cc.convert(b) for b in bullets], ensure_ascii=False),
            json.dumps(translated_links, ensure_ascii=False),
            now
        )
    )
    conn.commit()

def batch_summarize_with_gemini(client, conn, clusters):
    final_results = [None] * len(clusters)
    pending_items = []

    for idx, cluster in enumerate(clusters):
        links = [a["link"] for a in cluster["articles"]]
        cached = find_cached_summary(conn, links)
        if cached:
            valid_links = cached["translated_links"] or [
                {"source": a["source"], "title": ensure_traditional_chinese(a["title"]), "url": a["link"]}
                for a in cluster["articles"][:5]
            ]
            final_results[idx] = {
                "headline": cached["headline"],
                "bullets": cached["bullets"],
                "links": valid_links,
                "media_count": len(set(l["source"] for l in valid_links))
            }
        else:
            pending_items.append({
                "index": idx,
                "articles": cluster["articles"][:5]
            })

    if not pending_items:
        return final_results

    prompt_payload = [
        {
            "id": item["index"],
            "reports": [
                {
                    "report_idx": r_i,
                    "source": a["source"],
                    "title": ensure_traditional_chinese(a["title"]).replace('"', '「').replace('“', '「').replace('”', '」')
                }
                for r_i, a in enumerate(item["articles"])
            ]
        }
        for item in pending_items
    ]

    prompt = f"""
你是一位專業的台灣新聞總編輯。以下提供 {len(prompt_payload)} 組新聞事件群集：
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

請針對每一組事件（依 `id` 數字對應），嚴格以「台灣繁體中文（台灣慣用詞彙）」輸出：
1. `valid_report_indices`：檢查該組 `reports` 中所有標題，以第 0 則為核心，若混入完全無關的事件，請將無關新聞的 `report_idx` 剔除，僅回傳真正相關的整數陣列（如 `[0]` 或 `[0, 1]`）。
2. `headline`：將該事件改寫為一句話核心新聞標題（全繁體中文、禁止殘留任何日文假名或英文句子，約 18~25 字）。
3. `bullets`：根據標題資訊，改寫並提煉出 2 點關鍵重點或背景分析（全繁體中文，每點約 20~35 字，嚴格禁止直接照抄「媒體名稱：標題」）。

請嚴格輸出 JSON Array 格式：
[
  {{
    "id": 0,
    "valid_report_indices": [0],
    "headline": "事件核心總結標題",
    "bullets": ["重點摘要1", "重點摘要2"]
  }}
]
"""

    for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            text_cleaned = resp.text.strip()
            text_cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_cleaned, flags=re.MULTILINE).strip()
            raw_json = json.loads(text_cleaned)

            if isinstance(raw_json, dict):
                for v in raw_json.values():
                    if isinstance(v, list):
                        raw_json = v
                        break

            parsed_map = {}
            if isinstance(raw_json, list):
                for item in raw_json:
                    if isinstance(item, dict) and "id" in item:
                        try:
                            parsed_map[int(item["id"])] = item
                        except ValueError:
                            pass

            if not parsed_map:
                continue

            for item in pending_items:
                idx = item["index"]
                arts = item["articles"]

                if idx in parsed_map:
                    g_data = parsed_map[idx]
                    valid_idxs = g_data.get("valid_report_indices", list(range(len(arts))))
                    filtered_arts = [arts[i] for i in valid_idxs if isinstance(i, int) and 0 <= i < len(arts)]
                    if not filtered_arts:
                        filtered_arts = [arts[0]]

                    formatted_links = [
                        {"source": a["source"], "title": ensure_traditional_chinese(a["title"]), "url": a["link"]}
                        for a in filtered_arts
                    ]
                    all_valid_links = [a["link"] for a in filtered_arts]

                    headline = ensure_traditional_chinese(g_data.get("headline", filtered_arts[0]["title"]))
                    bullets = [ensure_traditional_chinese(b) for b in g_data.get("bullets", []) if b]
                    if not bullets:
                        bullets = [f"綜合 {filtered_arts[0]['source']} 等媒體報導，{headline}"]

                    save_summary_cache(conn, all_valid_links, headline, bullets, formatted_links)
                    final_results[idx] = {
                        "headline": headline,
                        "bullets": bullets,
                        "links": formatted_links,
                        "media_count": len(set(a["source"] for a in filtered_arts))
                    }
            break
        except Exception as e:
            print(f"Gemini ({model_name}) 呼叫失敗: {e}")
            time.sleep(4)

    for idx, res in enumerate(final_results):
        if res is None:
            arts = clusters[idx]["articles"][:5]
            zh_main_title = ensure_traditional_chinese(arts[0]["title"])
            final_results[idx] = {
                "headline": zh_main_title,
                "bullets": [f"{a['source']}：{ensure_traditional_chinese(a['title'])}" for a in arts[:3]],
                "links": [
                    {"source": a["source"], "title": ensure_traditional_chinese(a["title"]), "url": a["link"]}
                    for a in arts
                ],
                "media_count": len(set(a["source"] for a in arts))
            }

    return final_results

def cluster_and_rank(conn, embed_model, gemini_client, category, top_k=10):
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    c.execute("SELECT link, title, source FROM articles WHERE category = ? AND fetched_at >= ?", (category, cutoff))
    rows = c.fetchall()
    if not rows:
        return []

    # 分群前再次確保過濾掉任何無用標題並完成日/英翻中
    articles = []
    for r_link, r_title, r_source in rows:
        zh_t = ensure_traditional_chinese(strip_media_suffixes(r_title, is_google_news=True))
        if not is_junk_title(r_title, zh_t, r_source):
            articles.append({"link": r_link, "title": zh_t, "source": r_source})

    if not articles:
        return []

    if len(articles) == 1:
        raw_clusters = {0: [0]}
        dist_matrix = np.zeros((1, 1))
    else:
        titles = [a["title"] for a in articles]
        embeddings = embed_model.encode(titles, normalize_embeddings=True)
        dist_matrix = cosine_distances(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.20,
            metric="precomputed",
            linkage="average"
        ).fit(dist_matrix)

        raw_clusters = {}
        for idx, label in enumerate(clustering.labels_):
            raw_clusters.setdefault(label, []).append(idx)

    verified_groups = []
    for label, idx_list in raw_clusters.items():
        if len(idx_list) == 1:
            verified_groups.append([articles[idx_list[0]]])
            continue

        anchor_idx = idx_list[0]
        anchor_title = articles[anchor_idx]["title"]
        main_group = [articles[anchor_idx]]

        for member_idx in idx_list[1:]:
            member_title = articles[member_idx]["title"]
            cos_dist = dist_matrix[anchor_idx][member_idx]
            if is_truly_related(anchor_title, member_title, cos_dist):
                main_group.append(articles[member_idx])
            else:
                verified_groups.append([articles[member_idx]])

        verified_groups.append(main_group)

    ranked = []
    for group in verified_groups:
        unique_sources = {}
        for item in group:
            if item["source"] not in unique_sources:
                unique_sources[item["source"]] = item
        ranked.append({
            "unique_source_count": len(unique_sources),
            "articles": list(unique_sources.values()),
            "all_articles": group
        })

    ranked.sort(key=lambda x: (x["unique_source_count"], len(x["all_articles"])), reverse=True)
    top_clusters = ranked[:top_k]

    summarized_list = batch_summarize_with_gemini(gemini_client, conn, top_clusters)

    results = []
    for idx, (cluster, summary_data) in enumerate(zip(top_clusters, summarized_list), start=1):
        results.append({
            "rank": idx,
            "title": summary_data["headline"],
            "summary": summary_data["bullets"],
            "media_count": summary_data.get("media_count", cluster["unique_source_count"]),
            "links": summary_data["links"]
        })
    return results

def cleanup_old_data(conn):
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    c.execute("DELETE FROM articles WHERE fetched_at < ?", (cutoff,))
    c.execute("DELETE FROM summary_cache WHERE created_at < ?", (cutoff,))
    conn.commit()

def main():
    conn = init_db()
    new_articles, error_logs = fetch_all_feeds(conn)

    if new_articles:
        check_keywords(new_articles)

    if error_logs:
        err_msg = "⚠️ <b>RSS 媒體讀取異常報告</b>\n" + "\n".join(error_logs[:15])
        if len(error_logs) > 15:
            err_msg += f"\n...及其他 {len(error_logs)-15} 個錯誤"
        send_tg_message(err_msg)

    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    tw_top10 = cluster_and_rank(conn, embed_model, gemini_client, "tw", top_k=10)
    time.sleep(4)
    intl_top10 = cluster_and_rank(conn, embed_model, gemini_client, "intl", top_k=10)

    tw_tz = timezone(timedelta(hours=8))
    widget_data = {
        "updated_at": datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M"),
        "taiwan": tw_top10,
        "global": intl_top10
    }

    with open(WIDGET_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(widget_data, f, ensure_ascii=False, indent=2)

    cleanup_old_data(conn)
    conn.close()

if __name__ == "__main__":
    main()
