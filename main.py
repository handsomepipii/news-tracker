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
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from google import genai

GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()

DB_PATH = "news.db"
KEYWORDS_PATH = "keywords.json"
WIDGET_JSON_PATH = "widget_data.json"

cc = OpenCC('s2twp')

# 垃圾目錄文與無意義彙整標題黑名單（直接攔截韓聯社報紙頭條等目錄文）
JUNK_PATTERNS = [
    "主要報紙頭條", "報紙頭條新聞", "主要报纸头条", "新聞提要", "新闻提要","頭條新聞","各報重點",
    "今日頭條新聞", "每日新聞摘要", "新聞早報", "新聞晚報", "一周新聞", "一週新聞", "韓國歷史上的今天"
]

TW_FEEDS = {
    "中央社": "https://feeds.feedburner.com/cnaFirstNews",
    "公視新聞": "https://news.google.com/rss/search?q=site:news.pts.org.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "聯合新聞": "https://udn.com/rssfeed/news/2/6638?ch=news",
    "自由時報": "https://news.ltn.com.tw/rss/all.xml",
    "中時新聞": "https://news.google.com/rss/search?q=site:chinatimes.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "ETtoday": "https://feeds.feedburner.com/ettoday/realtime",
    "經濟日報": "https://money.udn.com/rssfeed/news/1001/5588",
    "數位時代": "https://news.google.com/rss/search?q=site:bnext.com.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "科技新報": "https://technews.tw/feed/",
    "報導者": "https://www.twreporter.org/a/rss2.xml",
    "關鍵評論網": "https://news.google.com/rss/search?q=site:thenewslens.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "轉角國際": "https://news.google.com/rss/search?q=site:global.udn.com+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "風傳媒": "https://news.google.com/rss/search?q=site:storm.mg+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "上報": "https://news.google.com/rss/search?q=site:upmedia.mg+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "鏡週刊": "https://www.mirrormedia.mg/rss/rss.xml",
    "鉅亨網": "https://news.google.com/rss/search?q=site:news.cnyes.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "新頭殼": "https://newtalk.tw/rss/all",
    "天下雜誌": "https://www.cw.com.tw/RSS/cw_content.xml",
    "TVBS新聞": "https://news.google.com/rss/search?q=site:news.tvbs.com.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "iThome": "https://www.ithome.com.tw/rss"
}

INTL_FEEDS = {
    "BBC中文": "https://feeds.bbci.co.uk/zhongwen/trad/rss.xml",
    "紐約時報中文": "https://news.google.com/rss/search?q=site:cn.nytimes.com+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "德國之聲": "https://rss.dw.com/rdf/rss-chi-all",
    "法廣RFI": "https://www.rfi.fr/tw/rss",
    "NHK新聞": "https://www.nhk.or.jp/rss/news/cat0.xml",
    "韓聯社": "https://cn.yna.co.kr/RSS/news.xml",
    "FT中文網": "https://www.ftchinese.com/rss/feed",
    "聯合國新聞": "https://news.un.org/feed/subscribe/zh/news/all/rss.xml",
    "CNN World": "http://rss.cnn.com/rss/edition_world.rss",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "CNBC World": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "Washington Post": "https://news.google.com/rss/search?q=site:washingtonpost.com+when:1d&hl=en-US&gl=US&ceid=US:en",
    "NPR World": "https://feeds.npr.org/1004/rss.xml",
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Financial Times": "https://www.ft.com/world?format=rss",
    "Sky News": "https://feeds.skynews.com/feeds/rss/world.xml"
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

def is_junk_title(title):
    return any(pat in title for pat in JUNK_PATTERNS)

def is_foreign_text(text):
    # 包含日文假名，或中文字少於 4 個字、或英文字母超過 10 個字，即判定需翻譯
    jp_chars = sum(1 for ch in text if '\u3040' <= ch <= '\u30ff')
    zh_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    en_chars = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    return jp_chars > 0 or zh_chars < 4 or (en_chars > 10 and en_chars > zh_chars)

def translate_to_zh_tw(text):
    """使用 Google 官方極速 GTX JSON 通道翻譯，穩定不擋 IP"""
    text = text.strip()
    if not text:
        return text
    # 移除常見無意義英文前綴
    text = re.sub(r'^(Live updates|Breaking news|Watch live|Video)\s*[:：\-]\s*', '', text, flags=re.IGNORECASE)
    if is_foreign_text(text):
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {"client": "gtx", "sl": "auto", "tl": "zh-TW", "dt": "t", "q": text}
            r = httpx.get(url, params=params, timeout=8.0)
            if r.status_code == 200:
                data = r.json()
                translated = "".join(seg[0] for seg in data[0] if seg[0])
                if translated:
                    text = translated
        except Exception as e:
            print(f"GTX 翻譯失敗: {e}")
    return cc.convert(text)

def clean_rss_title(raw_title, is_google_news=False):
    title = raw_title.strip()
    if is_google_news and " - " in title:
        title = title.rsplit(" - ", 1)[0].strip()
    return cc.convert(title)

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

    # 清除資料庫中既有的韓聯社報紙頭條等垃圾目錄文
    c.execute("SELECT link, title FROM articles")
    for link, title in c.fetchall():
        if is_junk_title(title):
            c.execute("DELETE FROM articles WHERE link = ?", (link,))

    # 清除舊快取中殘留外文或垃圾標題的紀錄
    c.execute("SELECT cluster_id, headline FROM summary_cache")
    for cid, headline in c.fetchall():
        if is_foreign_text(headline) or is_junk_title(headline):
            c.execute("DELETE FROM summary_cache WHERE cluster_id = ?", (cid,))

    conn.commit()
    return conn

def fetch_all_feeds(conn):
    c = conn.cursor()
    now = datetime.now(timezone.utc)
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

                    title = clean_rss_title(raw_title, is_google_news=is_gn)
                    if is_junk_title(title):
                        continue

                    c.execute("SELECT 1 FROM articles WHERE link = ?", (link,))
                    if not c.fetchone():
                        c.execute(
                            "INSERT INTO articles (link, title, source, category, fetched_at) VALUES (?, ?, ?, ?, ?)",
                            (link, title, source, category, now.isoformat())
                        )
                        new_articles.append({
                            "link": link,
                            "title": title,
                            "raw_title": raw_title,
                            "source": source,
                            "category": category
                        })
            except Exception as e:
                error_logs.append(f"• <b>{source}</b>: {type(e).__name__}")

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
        
        # 只要關鍵字出現在原始標題 OR 翻譯後標題，皆算命中
        matched = [kw for kw in keywords if kw.lower() in raw_t or kw.lower() in zh_t]
        if matched:
            zh_title = translate_to_zh_tw(a['title'])
            msg = (
                f"🚨 <b>關鍵字命中：{', '.join(matched)}</b>\n\n"
                f"📰 {zh_title}\n"
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
        if is_foreign_text(headline) or is_junk_title(headline):
            continue
        cached_set = set(json.loads(row_links_json))
        intersection = len(curr_set & cached_set)
        if curr_set == cached_set or (intersection >= 2) or (intersection / max(len(curr_set), 1) >= 0.5):
            return {
                "headline": cc.convert(headline),
                "bullets": [cc.convert(b) for b in json.loads(bullets_json)],
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
            final_results[idx] = {
                "headline": cached["headline"],
                "bullets": cached["bullets"],
                "links": cached["translated_links"] or [
                    {"source": a["source"], "title": translate_to_zh_tw(a["title"]), "url": a["link"]}
                    for a in cluster["articles"][:5]
                ]
            }
        else:
            # 送給 Gemini 前，先將入選 Top 10 的外文標題全部翻成繁體中文！
            translated_arts = []
            for a in cluster["articles"][:5]:
                translated_arts.append({
                    "source": a["source"],
                    "title": translate_to_zh_tw(a["title"]),
                    "link": a["link"]
                })
            pending_items.append({
                "index": idx,
                "articles": translated_arts
            })

    if not pending_items:
        return final_results

    prompt_payload = [
        {
            "id": item["index"],
            "reports": [{"source": a["source"], "title": a["title"]} for a in item["articles"]]
        }
        for item in pending_items
    ]

    prompt = f"""
你是一位專業的台灣新聞總編輯。以下提供 {len(prompt_payload)} 組新聞事件群集：
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

請針對每一組事件（依 `id` 數字對應），嚴格以「台灣繁體中文（台灣慣用詞彙）」輸出：
1. `headline`：用一句話精確總結該事件核心標題（全繁體中文，約 18~25 字，語意完整）。
2. `bullets`：用 2 點列出此事件的關鍵重點或背景細節（全繁體中文，每點約 20~35 字）。

請嚴格輸出 JSON Array 格式：
[
  {{
    "id": 0,
    "headline": "事件核心總結標題",
    "bullets": ["重點1", "重點2"]
  }}
]
"""

    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            raw_text = resp.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
            raw_json = json.loads(raw_text)
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

            for item in pending_items:
                idx = item["index"]
                arts = item["articles"]
                all_links = [a["link"] for a in clusters[idx]["articles"]]
                formatted_links = [{"source": a["source"], "title": a["title"], "url": a["link"]} for a in arts]

                if idx in parsed_map:
                    g_data = parsed_map[idx]
                    headline = translate_to_zh_tw(g_data.get("headline", arts[0]["title"]))
                    bullets = [translate_to_zh_tw(b) for b in g_data.get("bullets", [])]
                    if not bullets:
                        bullets = [f"{a['source']}：{a['title']}" for a in arts[:2]]

                    save_summary_cache(conn, all_links, headline, bullets, formatted_links)
                    final_results[idx] = {
                        "headline": headline,
                        "bullets": bullets,
                        "links": formatted_links
                    }
            break
        except Exception as e:
            print(f"Gemini ({model_name}) 呼叫失敗: {e}")
            time.sleep(3)

    # 最終安全備援（因為 item["articles"] 已經過 translate_to_zh_tw 翻譯，備援 100% 是繁體中文）
    pending_map = {item["index"]: item["articles"] for item in pending_items}
    for idx, res in enumerate(final_results):
        if res is None:
            arts = pending_map.get(idx) or [
                {"source": a["source"], "title": translate_to_zh_tw(a["title"]), "link": a["link"]}
                for a in clusters[idx]["articles"][:5]
            ]
            final_results[idx] = {
                "headline": arts[0]["title"],
                "bullets": [f"{a['source']}：{a['title']}" for a in arts[:3]],
                "links": [{"source": a["source"], "title": a["title"], "url": a["link"]} for a in arts]
            }

    return final_results

def cluster_and_rank(conn, embed_model, gemini_client, category, top_k=10):
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    c.execute("SELECT link, title, source FROM articles WHERE category = ? AND fetched_at >= ?", (category, cutoff))
    rows = c.fetchall()
    if not rows:
        return []

    articles = [{"link": r[0], "title": r[1], "source": r[2]} for r in rows if not is_junk_title(r[1])]
    if not articles:
        return []

    titles = [a["title"] for a in articles]
    embeddings = embed_model.encode(titles, normalize_embeddings=True)
    clustering = DBSCAN(eps=0.24, min_samples=1, metric="cosine").fit(embeddings)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(articles[idx])

    ranked = []
    for label, group in clusters.items():
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
            "media_count": cluster["unique_source_count"],
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
        send_tg_message(err_msg)

    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    tw_top10 = cluster_and_rank(conn, embed_model, gemini_client, "tw", top_k=10)
    time.sleep(3)
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
