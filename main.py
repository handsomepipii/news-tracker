import os
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta, timezone
import feedparser
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from google import genai

# 環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

DB_PATH = "news.db"
KEYWORDS_PATH = "keywords.json"
WIDGET_JSON_PATH = "widget_data.json"

# 台灣 20 家主流媒體 RSS
TW_FEEDS = {
    "中央社": "https://feeds.feedburner.com/cnaFirstNews",
    "公視新聞": "https://news.pts.org.tw/xml/rssnews.xml",
    "聯合新聞": "https://udn.com/rssfeed/news/2/6638?ch=news",
    "自由時報": "https://news.ltn.com.tw/rss/all.xml",
    "中時新聞": "https://www.chinatimes.com/rss/realtimenews.xml",
    "ETtoday": "https://feeds.feedburner.com/ettoday/realtime",
    "經濟日報": "https://money.udn.com/rssfeed/news/1001/5588",
    "數位時代": "https://www.bnext.com.tw/rss",
    "科技新報": "https://technews.tw/feed/",
    "報導者": "https://www.twreporter.org/a/rss2.xml",
    "關鍵評論網": "https://www.thenewslens.com/rss",
    "轉角國際": "https://global.udn.com/rssfeed/news/8662/8663",
    "風傳媒": "https://www.storm.mg/feeds/all",
    "上報": "https://www.upmedia.mg/rss/news_all.php",
    "鏡週刊": "https://www.mirrormedia.mg/rss/rss.xml",
    "鉅亨網": "https://news.cnyes.com/api/v1/news/rss/headline",
    "新頭殼": "https://newtalk.tw/rss/all",
    "天下雜誌": "https://www.cw.com.tw/RSS/cw_content.xml",
    "環境資訊中心": "https://e-info.org.tw/taxonomy/term/258/all/feed",
    "iThome": "https://www.ithome.com.tw/rss"
}

# 國際 20 家主流媒體 RSS
INTL_FEEDS = {
    "BBC中文": "https://feeds.bbci.co.uk/zhongwen/trad/rss.xml",
    "紐約時報中文": "https://cn.nytimes.com/rss/zh-hant.xml",
    "德國之聲": "https://rss.dw.com/rdf/rss-chi-all",
    "法廣RFI": "https://www.rfi.fr/tw/rss",
    "NHK華語": "https://www3.nhk.or.jp/nhkworld/zh/news/rss/",
    "韓聯社": "https://cn.yna.co.kr/RSS/news.xml",
    "FT中文網": "https://www.ftchinese.com/rss/feed",
    "聯合國新聞": "https://news.un.org/feed/subscribe/zh/news/all/rss.xml",
    "CNN World": "http://rss.cnn.com/rss/edition_world.rss",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "CNBC World": "https://search.cnbc.com/rs/search/combinedlist/view.xml?partnerId=wrss01&id=100727362",
    "Washington Post": "https://feeds.washingtonpost.com/rss/world",
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

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 儲存抓取過的文章
    c.execute('''CREATE TABLE IF NOT EXISTS articles (
                    link TEXT PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    category TEXT,
                    fetched_at TIMESTAMP
                )''')
    # 儲存已透過 Gemini 摘要過的事件快取（省 Token）
    c.execute('''CREATE TABLE IF NOT EXISTS summary_cache (
                    cluster_id TEXT PRIMARY KEY,
                    article_links TEXT,
                    headline TEXT,
                    bullets TEXT,
                    created_at TIMESTAMP
                )''')
    conn.commit()
    return conn

def fetch_all_feeds(conn):
    c = conn.cursor()
    now = datetime.now(timezone.utc)
    new_articles = []
    error_logs = []

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for category, feed_dict in [("tw", TW_FEEDS), ("intl", INTL_FEEDS)]:
        for source, url in feed_dict.items():
            try:
                resp = httpx.get(url, headers=headers, timeout=12.0, follow_redirects=True)
                if resp.status_code >= 400:
                    error_logs.append(f"• <b>{source}</b>: HTTP {resp.status_code}")
                    continue
                feed = feedparser.parse(resp.content)
                if not feed.entries:
                    error_logs.append(f"• <b>{source}</b>: 解析無文章 (可能格式變更)")
                    continue

                for entry in feed.entries[:20]: # 每家每次最多取最新20篇
                    link = entry.get("link", "").strip()
                    title = entry.get("title", "").strip()
                    if not link or not title:
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
        matched = [kw for kw in keywords if kw.lower() in a["title"].lower()]
        if matched:
            msg = (
                f"🚨 <b>關鍵字命中：{', '.join(matched)}</b>\n\n"
                f"📰 {a['title']}\n"
                f"🏢 來源：{a['source']}\n"
                f"🔗 <a href='{a['link']}'>點此閱讀報導</a>"
            )
            send_tg_message(msg)

def find_cached_summary(conn, current_links):
    """
    檢查過去 12 小時內是否已經有高度重疊的群集被 Gemini 摘要過。
    只要新群集與快取群集的文章重疊率 >= 50% (或至少 2 篇核心報導相同)，直接沿用摘要，不耗費 Token！
    """
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    c.execute("SELECT article_links, headline, bullets FROM summary_cache WHERE created_at >= ?", (cutoff,))
    rows = c.fetchall()

    curr_set = set(current_links)
    for row_links_json, headline, bullets_json in rows:
        cached_set = set(json.loads(row_links_json))
        intersection = len(curr_set & cached_set)
        # 若完全相同，或重疊超過一半，或至少有 2 篇相同報導歸在同一群
        if curr_set == cached_set or (intersection >= 2) or (intersection / max(len(curr_set), 1) >= 0.5):
            return {
                "headline": headline,
                "bullets": json.loads(bullets_json)
            }
    return None

def save_summary_cache(conn, current_links, headline, bullets):
    c = conn.cursor()
    sorted_links = sorted(list(set(current_links)))
    cluster_id = hashlib.md5("".join(sorted_links).encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        "INSERT OR REPLACE INTO summary_cache (cluster_id, article_links, headline, bullets, created_at) VALUES (?, ?, ?, ?, ?)",
        (cluster_id, json.dumps(sorted_links), headline, json.dumps(bullets, ensure_ascii=False), now)
    )
    conn.commit()

def summarize_with_gemini(client, conn, cluster):
    links = [a["link"] for a in cluster["articles"]]
    # 1. 先查快取，命中則跳過 API 呼叫
    cached = find_cached_summary(conn, links)
    if cached:
        print(f"[Cache Hit] 沿用既有摘要: {cached['headline']}")
        return cached

    # 2. 若為單篇新聞且為中文，甚至可直接用原標題作為備援，但為求品質統一交由 Gemini 濃縮
    titles_text = "\n".join([f"- {a['source']}: {a['title']}" for a in cluster["articles"][:8]])
    prompt = f"""
    以下是過去12小時內多家媒體對同一個新聞事件的報導標題：
    {titles_text}

    請以台灣繁體中文（若為外文新聞請直接翻譯為繁體中文）撰寫：
    1. headline：用一句話精確總結此事件核心（嚴格限制 25 字以內，供手機桌面小工具顯示）。
    2. bullets：用 2 到 3 點列出此事件的關鍵背景或重點細節。
    請嚴格輸出 JSON 格式：{{"headline": "...", "bullets": ["重點1", "重點2"]}}
    """
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        res_data = json.loads(resp.text)
        headline = res_data.get("headline", cluster["articles"][0]["title"][:25])
        bullets = res_data.get("bullets", ["點擊下方連結查看各家媒體完整報導"])
        save_summary_cache(conn, links, headline, bullets)
        return {"headline": headline, "bullets": bullets}
    except Exception as e:
        print(f"Gemini API 失敗，使用原始標題備援: {e}")
        return {
            "headline": cluster["articles"][0]["title"][:28],
            "bullets": [f"{a['source']}：{a['title']}" for a in cluster["articles"][:3]]
        }

def cluster_and_rank(conn, embed_model, gemini_client, category, top_k=10):
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    c.execute("SELECT link, title, source FROM articles WHERE category = ? AND fetched_at >= ?", (category, cutoff))
    rows = c.fetchall()
    if not rows:
        return []

    articles = [{"link": r[0], "title": r[1], "source": r[2]} for r in rows]
    titles = [a["title"] for a in articles]

    # 文字向量化與 DBSCAN 分群
    embeddings = embed_model.encode(titles, normalize_embeddings=True)
    clustering = DBSCAN(eps=0.24, min_samples=1, metric="cosine").fit(embeddings)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(articles[idx])

    ranked = []
    for label, group in clusters.items():
        # 以「獨立媒體來源數」排序，避免單一媒體洗稿
        unique_sources = {}
        for item in group:
            if item["source"] not in unique_sources:
                unique_sources[item["source"]] = item
        ranked.append({
            "unique_source_count": len(unique_sources),
            "articles": list(unique_sources.values()), # 每個媒體只留一篇代表作連結
            "all_articles": group
        })

    # 依獨立媒體數降序排列，取前 10 名
    ranked.sort(key=lambda x: (x["unique_source_count"], len(x["all_articles"])), reverse=True)
    top_clusters = ranked[:top_k]

    results = []
    for idx, cluster in enumerate(top_clusters, start=1):
        summary = summarize_with_gemini(gemini_client, conn, cluster)
        results.append({
            "rank": idx,
            "title": summary["headline"],
            "summary": summary["bullets"],
            "media_count": cluster["unique_source_count"],
            "links": [{"source": a["source"], "title": a["title"], "url": a["link"]} for a in cluster["articles"][:5]]
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

    # 1. 關鍵字即時推播
    if new_articles:
        check_keywords(new_articles)

    # 2. 若有媒體讀取失敗，推播至 Telegram 方便後續維修
    if error_logs:
        err_msg = "⚠️ <b>RSS 媒體讀取異常報告</b>\n" + "\n".join(error_logs[:15])
        if len(error_logs) > 15:
            err_msg += f"\n...及其他 {len(error_logs)-15} 個錯誤"
        send_tg_message(err_msg)

    # 3. 載入本地多語言 Embedding 模型與 Gemini
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    tw_top10 = cluster_and_rank(conn, embed_model, gemini_client, "tw", top_k=10)
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
