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

# 簡轉繁與多語系翻譯工具
cc = OpenCC('s2twp')
en_translator = GoogleTranslator(source='auto', target='zh-TW')

# 台灣主流與科技財經媒體 RSS (擴充科技與財經來源)
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
    # --- 台灣財經與產業媒體 ---
    "經濟日報": "https://money.udn.com/rssfeed/news/1001/5588",
    "工商時報": "https://news.google.com/rss/search?q=site:ctee.com.tw+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "鉅亨網": "https://news.google.com/rss/search?q=site:news.cnyes.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "MoneyDJ理財網": "https://news.google.com/rss/search?q=site:moneydj.com+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "天下雜誌": "https://www.cw.com.tw/RSS/cw_content.xml",
    "商業周刊": "https://news.google.com/rss/search?q=site:businessweekly.com.tw+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "今周刊": "https://news.google.com/rss/search?q=site:businesstoday.com.tw+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    "財訊": "https://news.google.com/rss/search?q=site:wealth.com.tw+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
}

# 國際主流與科技財經媒體 RSS (擴充全球財經、美股與硬體科技來源)
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
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
    "Nikkei Asia": "https://news.google.com/rss/search?q=site:asia.nikkei.com+when:1d&hl=en-US&gl=US&ceid=US:en",
    # --- 國際科技、AI 與半導體硬體 ---
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Wired": "https://www.wired.com/feed/rss",
    "VentureBeat": "https://venturebeat.com/feed/",
    "Tom's Hardware": "https://www.tomshardware.com/feeds/all",
    "Wccftech": "https://wccftech.com/feed/"
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

def is_mostly_english(text):
    # 偵測是否包含日文平假名 (\u3040-\u309f)、片假名 (\u30a0-\u30ff) 或外文為主
    jp_chars = sum(1 for ch in text if '\u3040' <= ch <= '\u30ff')
    zh_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    en_chars = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    return jp_chars > 0 or zh_chars == 0 or (en_chars > 12 and zh_chars < 4)

def ensure_traditional_chinese(text):
    text = text.strip()
    if not text:
        return text
    if is_mostly_english(text):
        try:
            translated = en_translator.translate(text)
            if translated:
                text = translated
        except Exception as e:
            print(f"翻譯備援失敗: {e}")
    return cc.convert(text)

def clean_rss_title(raw_title, is_google_news=False):
    title = raw_title.strip()
    if is_google_news and " - " in title:
        title = title.rsplit(" - ", 1)[0].strip()
    return ensure_traditional_chinese(title)

# --- 防誤分群第二道防線：字面關鍵詞交集檢查 ---
STOPWORDS_CHARS = set("的了在是與及或將為對從向被把讓給比跟和但而卻又也還更最很太都只就才已曾正要會能可好大中小上下前後裡外內高低長短新舊多少全各每某該此其之以為於與及等後前月日年時分秒週今明昨")

def extract_key_tokens(title):
    """提取標題中的中文雙字詞 (2-grams)、英文專有名詞與關鍵實字"""
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
    # 提取至少 2 字元以上的英文/數字專有名詞 (如 AI, TSMC, iPhone, Q3)
    en_tokens = set(re.findall(r'[a-zA-Z]{2,}[0-9]*', title.lower()))
    return bigrams, en_tokens, chars

def is_truly_related(title1, title2, cos_dist):
    """
    判斷兩則新聞是否真的在講同一件事：
    1. 若向量距離極近 (< 0.12)，直接放行
    2. 否則必須具備：至少 1 個共同中文雙字詞、或 1 個共同英文專有名詞、或至少 2 個共同核心中文字
    """
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

    c.execute("SELECT link, title FROM articles")
    for link, title in c.fetchall():
        if is_mostly_english(title):
            zh_title = ensure_traditional_chinese(title)
            c.execute("UPDATE articles SET title = ? WHERE link = ?", (zh_title, link))

    c.execute("SELECT cluster_id, headline FROM summary_cache")
    for cid, headline in c.fetchall():
        if is_mostly_english(headline):
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

                    c.execute("SELECT 1 FROM articles WHERE link = ?", (link,))
                    if not c.fetchone():
                        title = clean_rss_title(raw_title, is_google_news=is_gn)
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
        if is_mostly_english(headline):
            continue
        cached_set = set(json.loads(row_links_json))
        intersection = len(curr_set & cached_set)
        if curr_set == cached_set or (intersection >= 2 and intersection / max(len(curr_set), 1) >= 0.6):
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
            valid_links = cached["translated_links"] or [
                {"source": a["source"], "title": a["title"], "url": a["link"]}
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
                {"report_idx": r_i, "source": a["source"], "title": a["title"]}
                for r_i, a in enumerate(item["articles"])
            ]
        }
        for item in pending_items
    ]

    prompt = f"""
你是一位專業的台灣新聞總編輯。以下提供 {len(prompt_payload)} 組新聞事件群集：
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

請針對每一組事件（依 `id` 數字對應），嚴格以「台灣繁體中文（台灣慣用詞彙）」輸出：
1. `valid_report_indices`：檢查該組 `reports` 中所有標題，以第 0 則（或多數派主題）為核心，若裡面混入了「完全不同事件」的無關新聞，請將無關新聞的 `report_idx` 剔除，僅回傳真正相關的 `report_idx` 整數陣列（例如 `[0, 1]` 或 `[0]`）。
2. `headline`：針對相關報導，用一句話精確總結該事件核心標題（全繁體中文，約 18~25 字，語意完整不斷句）。
3. `bullets`：針對相關報導，用 2 點列出此事件的關鍵重點或背景細節（全繁體中文，每點約 20~35 字，禁止直接照抄「媒體名稱：標題」）。

請嚴格輸出 JSON Array 格式：
[
  {{
    "id": 0,
    "valid_report_indices": [0, 1],
    "headline": "事件核心總結標題",
    "bullets": ["重點1", "重點2"]
  }}
]
"""

    for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            raw_json = json.loads(resp.text)
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

                if idx in parsed_map:
                    g_data = parsed_map[idx]
                    valid_idxs = g_data.get("valid_report_indices", list(range(len(arts))))
                    filtered_arts = [arts[i] for i in valid_idxs if isinstance(i, int) and 0 <= i < len(arts)]
                    if not filtered_arts:
                        filtered_arts = [arts[0]]

                    formatted_links = [{"source": a["source"], "title": a["title"], "url": a["link"]} for a in filtered_arts]
                    all_valid_links = [a["link"] for a in filtered_arts]

                    headline = ensure_traditional_chinese(g_data.get("headline", filtered_arts[0]["title"]))
                    bullets = [ensure_traditional_chinese(b) for b in g_data.get("bullets", [])]
                    if not bullets:
                        bullets = [f"{a['source']}：{a['title']}" for a in filtered_arts[:2]]

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
            final_results[idx] = {
                "headline": ensure_traditional_chinese(arts[0]["title"]),
                "bullets": [f"{a['source']}：{ensure_traditional_chinese(a['title'])}" for a in arts[:3]],
                "links": [{"source": a["source"], "title": ensure_traditional_chinese(a["title"]), "url": a["link"]} for a in arts],
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

    articles = [{"link": r[0], "title": r[1], "source": r[2]} for r in rows]
    if len(articles) == 1:
        raw_clusters = {0: [0]}
        dist_matrix = np.zeros((1, 1))
    else:
        titles = [a["title"] for a in articles]
        embeddings = embed_model.encode(titles, normalize_embeddings=True)
        dist_matrix = cosine_distances(embeddings)

        # 第一道防線：使用平均連結階層分群 (Average Linkage)，徹底消滅 DBSCAN 單點連鎖誤判
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.20,
            metric="precomputed",
            linkage="average"
        ).fit(dist_matrix)

        raw_clusters = {}
        for idx, label in enumerate(clustering.labels_):
            raw_clusters.setdefault(label, []).append(idx)

    # 第二道防線：字面雙字詞/實體交集驗證，將誤混入的無關新聞拆出
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
                # 若字面毫無關聯，將其拆回獨立單篇事件，絕不硬湊
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
