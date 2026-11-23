import html
import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ArthaPulse", layout="wide")
DEFAULT = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "NVDA", "AAPL", "MSFT", "TSLA", "GC=F"]
RANGES = {"1M": 21, "3M": 63, "1Y": 252, "5Y": 1260}
ss = st.session_state
ss.setdefault("watch", DEFAULT.copy())
ss.setdefault("cash", 100000.0)
ss.setdefault("pos", {})
ss.setdefault("hist", [])


def is_inr(t):
    return t.endswith((".NS", ".BO"))


def sym(t):
    return "₹" if is_inr(t) else "$"


@st.cache_data(ttl=300)
def load(t, period="5y"):
    try:
        df = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False)
        return df.dropna(subset=["Close"]) if not df.empty else df
    except Exception:
        return pd.DataFrame()


def usdinr():
    df = load("USDINR=X", "5d")
    return float(df.Close.iloc[-1]) if len(df) else 85.0


def add_ind(df):
    d = df.copy()
    c = d.Close
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["SMA50"] = c.rolling(50).mean()
    delta = c.diff()
    g = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + g / l)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["HIST"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([d.High - d.Low, (d.High - c.shift()).abs(), (d.Low - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return d


def patterns(d):
    o, h, l, c = d.Open, d.High, d.Low, d.Close
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    p = {
        "Bullish Engulfing": ((c.shift() < o.shift()) & (c > o) & (c >= o.shift()) & (o <= c.shift()), 1),
        "Bearish Engulfing": ((c.shift() > o.shift()) & (c < o) & (o >= c.shift()) & (c <= o.shift()), -1),
        "Hammer": (((lower >= 2 * body) & (upper <= 0.5 * body) & (body / rng > 0.1) & (c < d.EMA20)), 1),
        "Breakout (20d high)": (c > h.rolling(20).max().shift(), 1),
        "Breakdown (20d low)": (c < l.rolling(20).min().shift(), -1),
        "Doji": (body / rng < 0.1, 0),
    }
    return {k: (v.fillna(False), s) for k, (v, s) in p.items()}


def backtest(d, series, direction):
    fwd = d.Close.shift(-5) / d.Close - 1
    m = series & fwd.notna()
    n = int(m.sum())
    if direction == 0 or n < 10:
        return n, None, None
    f = fwd[m]
    win = (f > 0).mean() if direction > 0 else (f < 0).mean()
    return n, win * 100, f.mean() * 100


def tech_score(d):
    r = d.iloc[-1]
    s = 25 if r.RSI < 30 else -25 if r.RSI > 70 else 0
    s += 25 if r.HIST > 0 else -25
    s += 25 if r.Close > r.EMA20 else -25
    s += 25 if r.EMA20 > r.SMA50 else -25
    return s


@st.cache_data(ttl=600)
def get_news(t):
    q = "gold price" if t == "GC=F" else t.replace(".NS", "").replace(".BO", "") + " stock"
    url = "https://news.google.com/rss/search?q=" + requests.utils.quote(q) + "&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        return [{"title": html.unescape(e.title), "link": e.link, "time": e.get("published", ""),
                 "src": e.get("source", {}).get("title", "")} for e in feed.entries[:8]]
    except Exception:
        return []


POS = ("surge", "gain", "rise", "beat", "profit", "growth", "record", "upgrade", "jump", "rally", "strong", "buy", "rebound")
NEG = ("fall", "drop", "loss", "miss", "downgrade", "slump", "crash", "probe", "fraud", "weak", "sell", "decline", "cut", "fine")


@st.cache_data(ttl=900)
def sentiment(titles, key, model):
    if key:
        try:
            prompt = ("For each numbered headline reply with exactly one word per line: Positive, Negative or Neutral. "
                      "No other text.\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles)))
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              params={"key": key}, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            out = [x.strip().capitalize() for x in txt.splitlines() if x.strip()]
            out = [x if x in ("Positive", "Negative", "Neutral") else "Neutral" for x in out]
            if len(out) == len(titles):
                return out, "Gemini"
        except Exception:
            pass
    res = []
    for t in titles:
        tl = t.lower()
        sc = sum(w in tl for w in POS) - sum(w in tl for w in NEG)
        res.append("Positive" if sc > 0 else "Negative" if sc < 0 else "Neutral")
    return res, "Keyword (basic)"


def final_signal(ts, ns):
    c = 0.7 * ts + 0.3 * ns
    if ts * ns < 0 and abs(ts) >= 25 and abs(ns) >= 25:
        return "Careful (technical aur news ulte)", c
    return ("Bullish" if c >= 25 else "Bearish" if c <= -25 else "Neutral"), c


def detail(t, key, model):
    df = load(t, "5y")
    if len(df) < 60:
        st.error("Data nahi aaya ya bahut kam hai. Baad me try karo.")
        return
    d = add_ind(df)
    r = d.iloc[-1]
    cs = sym(t)
    st.caption(f"Last data: {d.index[-1].strftime('%d %b %Y')} | Yahoo data delayed ho sakta hai")
    rng = st.radio("Range", list(RANGES), horizontal=True, index=1)
    v = d.tail(RANGES[rng])
    fig = go.Figure([go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close, name="Price"),
                     go.Scatter(x=v.index, y=v.EMA20, name="EMA20"), go.Scatter(x=v.index, y=v.SMA50, name="SMA50")])
    fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.bar_chart(v.Volume, height=120)

    sup, res = d.Low.tail(60).min(), d.High.tail(60).max()
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI (14)", f"{r.RSI:.1f}")
    c2.metric("MACD hist", f"{r.HIST:+.2f}")
    c3.metric("Stop-loss (2x ATR)", f"{cs}{r.Close - 2 * r.ATR:,.2f}")
    st.write(f"Support (60d): **{cs}{sup:,.2f}** | Resistance (60d): **{cs}{res:,.2f}**")

    st.subheader("Patterns aur 5-saal backtest (agle 5 din)")
    rows = []
    for name, (ser, dr) in patterns(d).items():
        n, win, avg = backtest(d, ser, dr)
        rows.append({"Pattern": name, "Aaj bana?": "Haan" if ser.iloc[-1] else "-", "Formations": n,
                     "Win rate %": "Not enough data" if win is None else round(win, 1),
                     "Avg 5d return %": "-" if avg is None else round(avg, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("News")
    items = get_news(t)
    if not items:
        st.warning("News nahi mili.")
    labels, how = sentiment(tuple(i["title"] for i in items), key, model) if items else ([], "-")
    for i, lab in zip(items, labels):
        em = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}[lab]
        st.markdown(f"{em} [{i['title']}]({i['link']})  \n<small>{i['src']} | {i['time']}</small>", unsafe_allow_html=True)
    ns = int(round(np.mean([{"Positive": 1, "Negative": -1}.get(x, 0) for x in labels]) * 100)) if labels else 0
    ts = tech_score(d)
    sig, comb = final_signal(ts, ns)
    st.subheader(f"Final Signal: {sig}")
    st.write(f"Technical score **{ts}**, News score **{ns}** ({how}), Combined **{comb:.0f}** (-100 se +100). "
             "Ye probability nahi hai, sirf indicators ka summary hai.")


st.title("📈 ArthaPulse")
with st.sidebar:
    new = st.text_input("Ticker add karo (jaise SBIN.NS, AMZN)")
    if st.button("Add") and new.strip():
        tk = new.strip().upper()
        if load(tk, "1mo").empty:
            st.error("Ye ticker nahi mila.")
        elif tk not in ss.watch:
            ss.watch.append(tk)
    key = st.text_input("Gemini API key (optional, better news sentiment)", type="password")
    model = st.text_input("Gemini model", "gemini-flash-latest")
    st.caption("Financial advice nahi hai, sirf educational tool. Data 15 min delayed ho sakta hai.")

tabs = st.tabs(["Watchlist", "Detail", "Screener", "Paper Trading", "Risk Calc"])

with tabs[0]:
    rows = []
    for t in ss.watch:
        df = load(t, "1mo")
        if len(df) < 2:
            rows.append({"Ticker": t, "Last": "Data nahi", "Prev close": "-", "Change %": "-", "Date": "-"})
            continue
        last, prev = df.Close.iloc[-1], df.Close.iloc[-2]
        rows.append({"Ticker": t, "Last": f"{sym(t)}{last:,.2f}", "Prev close": f"{sym(t)}{prev:,.2f}",
                     "Change %": f"{(last / prev - 1) * 100:+.2f}", "Date": df.index[-1].strftime("%d %b")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

with tabs[1]:
    detail(st.selectbox("Stock", ss.watch, key="det"), key, model)

with tabs[2]:
    f = st.radio("Filter", ["All", "Bullish", "Bearish"], horizontal=True)
    out = []
    for t in ss.watch:
        df = load(t, "1y")
        if len(df) < 60:
            continue
        d = add_ind(df)
        found = [(n, s) for n, (ser, s) in patterns(d).items() if ser.iloc[-1] and n != "Doji"]
        rsi = d.RSI.iloc[-1]
        if rsi < 30:
            found.append(("RSI oversold", 1))
        if rsi > 70:
            found.append(("RSI overbought", -1))
        for n, s in found:
            if f == "All" or (f == "Bullish" and s > 0) or (f == "Bearish" and s < 0):
                out.append({"Ticker": t, "Signal": n, "Direction": "Bullish" if s > 0 else "Bearish"})
    st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True) if out else st.info("Aaj koi signal nahi bana.")

with tabs[3]:
    fx = usdinr()
    st.caption(f"Virtual ₹1,00,000. US stocks USD/INR {fx:.2f} pe convert hote hain. Page refresh par data reset hota hai.")
    t = st.selectbox("Stock", ss.watch, key="pt")
    dfp = load(t, "1mo")
    if dfp.empty:
        st.error("Price nahi mila.")
    else:
        pi = float(dfp.Close.iloc[-1]) * (1 if is_inr(t) else fx)
        qty = st.number_input("Quantity", 1, 100000, 1)
        b, s_ = st.columns(2)
        q, a = ss.pos.get(t, (0, 0.0))
        if b.button("Buy"):
            cost = qty * pi
            if cost > ss.cash:
                st.error("Cash kam hai.")
            else:
                ss.pos[t] = (q + qty, (q * a + cost) / (q + qty))
                ss.cash -= cost
                ss.hist.append(("BUY", t, int(qty), round(pi, 2)))
        if s_.button("Sell"):
            if qty > q:
                st.error("Itni quantity holding me nahi hai.")
            else:
                ss.cash += qty * pi
                ss.hist.append(("SELL", t, int(qty), round(pi, 2)))
                if q - qty == 0:
                    del ss.pos[t]
                else:
                    ss.pos[t] = (q - qty, a)
        total, rows = ss.cash, []
        for tk, (qq, aa) in ss.pos.items():
            dd = load(tk, "1mo")
            now = float(dd.Close.iloc[-1]) * (1 if is_inr(tk) else fx) if len(dd) else aa
            total += qq * now
            rows.append({"Ticker": tk, "Qty": qq, "Avg ₹": round(aa, 2), "Now ₹": round(now, 2), "P&L ₹": round((now - aa) * qq, 2)})
        st.metric("Cash", f"₹{ss.cash:,.2f}")
        st.metric("Total value", f"₹{total:,.2f}", f"{total - 100000:+,.2f}")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if ss.hist:
            st.write("Trade history", pd.DataFrame(ss.hist, columns=["Side", "Ticker", "Qty", "Price ₹"]))

with tabs[4]:
    cap = st.number_input("Capital", 1.0, value=100000.0)
    rk = st.number_input("Risk %", 0.1, 100.0, 1.0)
    en = st.number_input("Entry price", 0.01, value=100.0)
    sl = st.number_input("Stop-loss", 0.01, value=95.0)
    if en != sl:
        risk_amt = cap * rk / 100
        qty = int(risk_amt // abs(en - sl))
        tgt = en + 2 * (en - sl)
        st.metric("Quantity", qty)
        st.write(f"Total risk: **{risk_amt:,.2f}** | Target (1:2): **{tgt:,.2f}** | Position size: **{qty * en:,.2f}**")
    else:
        st.warning("Entry aur stop-loss alag hone chahiye.")
