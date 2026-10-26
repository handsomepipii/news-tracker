import os
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

# 環境變數
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

DB_PATH = "news.db"
KEYWORDS_PATH = "keywords.json"
WIDGET_JSON_PATH = "widget_data.json"

# 初始化簡轉繁轉換器 (s2twp: 簡體到台灣正體並轉換慣用詞彙)
cc = OpenCC('s2twp')

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
    # 相容舊資料表結構：若無 translated_links 欄位則自動新增
    try:
        c.execute("ALTER TABLE summary_cache ADD COLUMN translated_links TEXT")
    except sqlite3.OperationalError:
        pass
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
                    error_logs.append(f"• <b>{source}</b>: 解析無文章")
                    continue

                for entry in feed.entries[:20]:
                    link = entry.get("link", "").strip()
                    raw_title = entry.get("title", "").strip()
                    if not link or not raw_title:
                        continue
                    
                    # 入庫前統一將簡體中文轉為台灣繁體中文
                    title = cc.convert(raw_title)

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

def has_english(text):
    # 判斷標題是否包含大量英文字母（用於過濾之前快取失敗的英文標題）
    ascii_letters = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    return ascii_letters > 8

def find_cached_summary(conn, current_links):
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    c.execute("SELECT article_links, headline, bullets, translated_links FROM summary_cache WHERE created_at >= ?", (cutoff,))
    rows = c.fetchall()

    curr_set = set(current_links)
    for row_links_json, headline, bullets_json, trans_links_json in rows:
        # 若快取中的標題含有未翻譯的英文，視為無效快取，重新交由 Gemini 翻譯
        if has_english(headline):
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
    """
    將需要摘要與翻譯的多個新聞群集「打包成單一請求」送給 Gemini，
    避免逐筆呼叫觸發 429 RPM 速率限制，同時確保所有外文標題 100% 翻譯為繁體中文！
    """
    final_results = [None] * len(clusters)
    pending_items = []

    for idx, cluster in enumerate(clusters):
        links = [a["link"] for a in cluster["articles"]]
        cached = find_cached_summary(conn, links)
        if cached:
            print(f"[Cache Hit] 沿用繁中摘要: {cached['headline']}")
            final_results[idx] = {
                "headline": cached["headline"],
                "bullets": cached["bullets"],
                "links": cached["translated_links"] or [
                    {"source": a["source"], "title": cc.convert(a["title"]), "url": a["link"]}
                    for a in cluster["articles"][:5]
                ]
            }
        else:
            pending_items.append({
                "index": idx,
                "articles": cluster["articles"][:5]
            })

    if not pending_items:
        return final_results

    # 準備批次 Prompt 資料
    prompt_payload = []
    for item in pending_items:
        prompt_payload.append({
            "id": item["index"],
            "reports": [{"source": a["source"], "title": a["title"]} for a in item["articles"]]
        })

    prompt = f"""
你是一位專業的台灣新聞總編輯。以下提供 {len(prompt_payload)} 組新聞事件群集（包含多家國內外媒體的報導標題）：
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

請針對每一組事件（依 `id` 對應），嚴格以「台灣繁體中文（台灣慣用詞彙，如：川普、歐巴馬、魯拉、馬斯克、北韓）」輸出：
1. `headline`：用一句話精確總結該事件核心標題（**必須為全繁體中文，嚴格禁止出現英文句子**，字數控制在 18~24 字內，完整不斷句）。
2. `bullets`：用 2 點列出此事件的關鍵重點或背景細節（**全繁體中文**，每點約 20~35 字）。
3. `translated_titles`：將該組 `reports` 中的每一則原始標題翻譯為「台灣繁體中文」（依原本順序排列的字串陣列）。

請嚴格輸出 JSON Array 格式，範例如下：
[
  {{
    "id": 0,
    "headline": "巴西大選魯拉險勝連任，展開第四度總統任期",
    "bullets": [
      "巴西左翼總統魯拉於第二輪決選中擊敗右翼對手波索納洛。",
      "多國領袖致電祝賀，市場關注後續財政與環保政策走向。"
    ],
    "translated_titles": [
      "魯拉於歷史性決選勝出，第四度當選巴西總統",
      "巴西左翼總統魯拉成功連任第四個任期"
    ]
  }}
]
"""

    # 具備自動重試機制的 API 呼叫
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            parsed_list = json.loads(resp.text)
            parsed_map = {item["id"]: item for item in parsed_list if isinstance(item, dict) and "id" in item}

            for item in pending_items:
                idx = item["index"]
                arts = item["articles"]
                all_links = [a["link"] for a in clusters[idx]["articles"]]

                if idx in parsed_map:
                    g_data = parsed_map[idx]
                    headline = cc.convert(g_data.get("headline", arts[0]["title"]))
                    bullets = [cc.convert(b) for b in g_data.get("bullets", [])]
                    trans_titles = g_data.get("translated_titles", [])

                    formatted_links = []
                    for i, a in enumerate(arts):
                        t_title = cc.convert(trans_titles[i]) if i < len(trans_titles) else cc.convert(a["title"])
                        formatted_links.append({
                            "source": a["source"],
                            "title": t_title,
                            "url": a["link"]
                        })

                    save_summary_cache(conn, all_links, headline, bullets, formatted_links)
                    final_results[idx] = {
                        "headline": headline,
                        "bullets": bullets,
                        "links": formatted_links
                    }
            break
        except Exception as e:
            print(f"Gemini 批次呼叫第 {attempt+1} 次失敗: {e}")
            time.sleep(6 * (attempt + 1))

    # 若仍有未成功項目之最終安全備援
    for idx, res in enumerate(final_results):
        if res is None:
            arts = clusters[idx]["articles"][:5]
            final_results[idx] = {
                "headline": cc.convert(arts[0]["title"][:24]),
                "bullets": [f"{a['source']}：{cc.convert(a['title'])}" for a in arts[:2]],
                "links": [{"source": a["source"], "title": cc.convert(a["title"]), "url": a["link"]} for a in arts]
            }

    return final_results

def cluster_and_rank(conn, embed_model, gemini_client, category, top_k=10):
    c = conn.cursor()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    c.execute("SELECT link, title, source FROM articles WHERE category = ? AND fetched_at >= ?", (category, cutoff))
    rows = c.fetchall()
    if not rows:
        return []

    articles = [{"link": r[0], "title": cc.convert(r[1]), "source": r[2]} for r in rows]
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

    # 一次性批次交由 Gemini 進行繁中統整與翻譯
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
        if len(error_logs) > 15:
            err_msg += f"\n...及其他 {len(error_logs)-15} 個錯誤"
        send_tg_message(err_msg)

    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    tw_top10 = cluster_and_rank(conn, embed_model, gemini_client, "tw", top_k=10)
    time.sleep(3) # 緩衝 3 秒再處理國際新聞
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
