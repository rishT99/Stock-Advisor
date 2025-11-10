"""
Stock Advisor Bot — India-first (v3.2, Fundamentals Enhanced)

What's new vs v3.1:
- Fundamentals layer (Yahoo Finance .info) with a compact scoring model
- Blended score: total = 0.8 * technical + 0.2 * fundamentals_norm
- Rationale includes fundamentals details + subscore fields
- Volume burst threshold relaxed (1.2× MA20 instead of 1.5×)
- JSON-safe responses and global error handlers retained

Run:
  python -m venv .venv && .venv\Scripts\activate
  pip install -r requirements.txt
  uvicorn app:app --reload --port 8000

Open: http://localhost:8000/dashboard
"""
from __future__ import annotations

import logging
import os, re, math
from difflib import get_close_matches
from datetime import datetime
from typing import Optional, Literal, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# -------- Pretty logging --------
try:
    from colorama import init as colorama_init, Fore, Style  # type: ignore
    colorama_init()
    C_OK = Fore.GREEN + "✓" + Style.RESET_ALL
    C_WARN = Fore.YELLOW + "!" + Style.RESET_ALL
    C_ERR = Fore.RED + "✗" + Style.RESET_ALL
    C_INFO = Fore.CYAN + "→" + Style.RESET_ALL
except Exception:
    C_OK, C_WARN, C_ERR, C_INFO = "✓", "!", "✗", "→"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("stockadvisor")

# -------- Optional imports (guarded) --------
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import ta  # type: ignore
except Exception:
    ta = None

try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None

app = FastAPI(title="Stock Advisor Bot — India v3.2")
templates = Jinja2Templates(directory="templates")

# ==========================
# JSON SANITIZER + HANDLERS
# ==========================
def _num_sane(x):
    """Return a JSON-safe numeric or None."""
    try:
        if x is None:
            return None
        # numpy scalars
        if isinstance(x, (np.floating, np.integer)):
            x = float(x) if isinstance(x, np.floating) else int(x)
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
        return x
    except Exception:
        return None

def _to_jsonable(obj):
    """Recursively convert objects to JSON-safe structures."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float, np.floating, np.integer)):
        return _num_sane(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, datetime)):
        try:
            ts = obj.tz_convert("UTC") if getattr(obj, "tzinfo", None) else obj
        except Exception:
            ts = obj
        try:
            return pd.to_datetime(ts, utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            return str(ts)
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if isinstance(obj, (np.ndarray,)):
        return [_to_jsonable(v) for v in obj.tolist()]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)

def json_ok(payload: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status, content=_to_jsonable(payload))

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    log.warning(f"{C_WARN} HTTPException {exc.status_code}: {exc.detail}")
    return json_ok({"ok": False, "error": str(exc.detail)}, status=exc.status_code)

@app.exception_handler(Exception)
async def unhandled_exc(request: Request, exc: Exception):
    log.exception(f"{C_ERR} Unhandled error: {exc}")
    return json_ok({"ok": False, "error": "internal_error"}, status=500)

# -------- In-memory news store --------
# Each item: {"symbol": "RELIANCE" or "*" for market, "headline": str, "ts": pd.Timestamp(UTC-aware)}
NEWS_DB: List[Dict[str, Any]] = []

# -------- Models --------
class AdviceRequest(BaseModel):
    symbol: str = Field(..., description="Ticker. 'RELIANCE' auto-uses .NS; use .BO for BSE")
    goal: Optional[Literal["short_term", "swing", "long_term"]] = Field(None, description="If omitted, inferred")
    wallet: float = Field(..., ge=0)
    risk_level: Literal["low", "medium", "high"] = "medium"
    data_csv: Optional[str] = Field(None, description="Optional local OHLCV CSV if no internet.")
    min_confluence: int = Field(2, description="Required green lights among trend/momentum/volume/sentiment")
    max_daily_risk_frac: float = Field(0.02, description="Cap total risk opened today, e.g. 2% of wallet")

class Target(BaseModel):
    label: str
    price: float

class AdviceResponse(BaseModel):
    symbol: str
    as_of: str
    decision: Literal["BUY", "HOLD", "AVOID"]
    rationale: Dict[str, Any]
    entry_price: float
    stop_loss: float
    risk_per_share: float
    targets: List[Target]
    position_qty: int
    max_alloc_inr: float
    est_reward_risk: float

class ExplainRequest(BaseModel):
    symbol: str
    on_date: Optional[str] = Field(None, description="YYYY-MM-DD; if None, uses last row")
    data_csv: Optional[str] = None

class IngestNewsItem(BaseModel):
    symbol: Optional[str] = Field(None, description="e.g., RELIANCE; use * for market-wide")
    headline: str
    ts: Optional[str] = Field(None, description="ISO timestamp; default now")

class IngestNewsRequest(BaseModel):
    items: List[IngestNewsItem]

class AutoNewsRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    max_per_symbol: int = 8

class ScanRequest(BaseModel):
    symbols: List[str]
    wallet: float
    risk_level: Literal["low","medium","high"] = "medium"
    min_confluence: int = 2
    max_daily_risk_frac: float = 0.02

# ---- UI Log Buffer (ring) ----
class UIBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 2000):
        super().__init__()
        self.capacity = capacity
        self.buf: List[str] = []

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.buf.append(msg)
        if len(self.buf) > self.capacity:
            self.buf = self.buf[-self.capacity:]

UI_LOG_HANDLER = UIBufferHandler(capacity=2000)
UI_LOG_HANDLER.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(UI_LOG_HANDLER)

# -------- Symbol universe / normalization --------
SYMBOL_UNI: Dict[str, str] = {}
STOPWORDS = {"LTD","LIMITED","CO","COMPANY","PLC","PVT","INC","AND","OF","THE"}

def canonicalize_name(name: str) -> str:
    s = (name or "").upper()
    s = s.replace("&"," AND ").replace("."," ").replace(","," ")
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    parts = [p for p in s.split() if p and p not in STOPWORDS]
    return " ".join(parts)

def load_symbol_universe():
    path = os.path.join("data", "symbols_india.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            cnt = 0
            for _, r in df.iterrows():
                nm = str(r.get("name","")).strip().upper()
                sym = str(r.get("symbol","")).strip().upper()
                if nm and sym:
                    SYMBOL_UNI[nm] = sym
                    cnt += 1
            log.info(f"{C_OK} Loaded {cnt} symbols from data/symbols_india.csv")
        except Exception as e:
            log.info(f"{C_WARN} Could not load symbols_india.csv: {e}")

def _best_symbol_guess_from_universe(name: str) -> Optional[str]:
    if not SYMBOL_UNI:
        return None
    canon = canonicalize_name(name)
    candidates = get_close_matches(canon, list(SYMBOL_UNI.keys()), n=1, cutoff=0.86)
    if candidates:
        return SYMBOL_UNI[candidates[0]]
    return None

load_symbol_universe()

def normalize_symbol_for_india(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if s.endswith(".NS") or s.endswith(".BO") or s.startswith("^"):
        return s
    uni = _best_symbol_guess_from_universe(s)
    if uni:
        return f"{uni}.NS"
    canon = canonicalize_name(s)
    nospace = canon.replace(" ", "")
    return f"{nospace}.NS"

# -------- Helpers --------
def _utc_aware(ts_like) -> pd.Timestamp:
    if isinstance(ts_like, pd.Timestamp):
        if ts_like.tzinfo is None:
            return ts_like.tz_localize("UTC")
        return ts_like.tz_convert("UTC")
    try:
        return pd.to_datetime(ts_like, utc=True)
    except Exception:
        return pd.Timestamp.now(tz="UTC")

def load_ohlcv(symbol: str, csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        log.info(f"{C_INFO} Loading OHLCV from CSV for {symbol}...")
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().capitalize() for c in df.columns]
        if "Date" not in df.columns:
            log.info(f"{C_ERR} CSV missing 'Date' column")
            raise HTTPException(400, "CSV must include a 'Date' column")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        log.info(f"{C_OK} Loaded {len(df)} rows from CSV")
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if yf is None:
        log.info(f"{C_ERR} yfinance not available — cannot fetch {symbol}")
        raise HTTPException(400, "yfinance not available. Provide data_csv.")

    try:
        ysym = normalize_symbol_for_india(symbol)
        log.info(f"{C_INFO} Loading OHLCV from Yahoo for {ysym}...")
        y = yf.Ticker(ysym)
        df = y.history(period="400d")
        if len(df) < 260:
            df = y.history(period="max")

        if df.empty or len(df) < 20:
            base = ysym[:-3] if (ysym.endswith(".NS") or ysym.endswith(".BO")) else ysym
            candidates: List[str] = []
            uni_guess = _best_symbol_guess_from_universe(symbol)
            if uni_guess:
                candidates.append(uni_guess + ".NS")
            candidates.append(base.replace(" ", "") + (".NS" if not base.endswith((".NS",".BO")) else ""))
            if not base.endswith(".BO"):
                candidates.append(base.replace(" ", "") + ".BO")

            for cand in [c for c in dict.fromkeys(candidates) if c and c != ysym]:
                try:
                    log.info(f"{C_WARN} No/low data for {ysym}. Retrying with {cand} ...")
                    y2 = yf.Ticker(cand)
                    df2 = y2.history(period="400d")
                    if len(df2) < 260:
                        df2 = y2.history(period="max")
                    if not df2.empty:
                        ysym = cand
                        df = df2
                        break
                except Exception:
                    pass

        if df.empty:
            log.info(f"{C_WARN} No data for {ysym}")
            raise HTTPException(404, f"No data for {ysym}")

        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        log.info(f"{C_OK} Got {len(df)} rows of OHLCV for {ysym}")
        return df
    except HTTPException:
        raise
    except Exception as e:
        log.info(f"{C_ERR} Failed to fetch {symbol}: {e}")
        raise HTTPException(400, f"Failed to fetch data for {symbol}: {e}")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"{C_INFO} Computing features (ATR, SMAs, ROC, Volume, RSI)...")
    d = df.copy()
    tr1 = (d["High"] - d["Low"]).abs()
    tr2 = (d["High"] - d["Close"].shift()).abs()
    tr3 = (d["Low"] - d["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d["ROC10"] = d["Close"].pct_change(10)
    d["ROC50"] = d["Close"].pct_change(50)
    d["VolMA20"] = d["Volume"].rolling(20).mean()
    # relaxed threshold: 1.2x MA20
    d["VolBurst"] = (d["Volume"] > 1.2 * d["VolMA20"]).astype(int)
    if ta is not None:
        try:
            d["RSI14"] = ta.momentum.RSIIndicator(d["Close"], window=14).rsi()
        except Exception:
            d["RSI14"] = np.nan
    else:
        d["RSI14"] = np.nan
    d = d.dropna()
    log.info(f"{C_OK} Features ready (rows={len(d)})")
    return d

# -------- Market Regime (NIFTY) --------
def get_nifty_regime() -> Dict[str, Any]:
    sym = "^NSEI"
    if yf is None:
        return {"symbol": sym, "ok": False, "reason": "yfinance not available"}
    try:
        log.info(f"{C_INFO} Checking NIFTY 50 regime...")
        df = yf.Ticker(sym).history(period="max")
        if df.empty:
            return {"symbol": sym, "ok": False, "reason": "no data"}
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df["SMA200"] = df["Close"].rolling(200).mean()
        row = df.iloc[-1]
        regime_up = bool(row["Close"] > row["SMA200"])
        ret5d = float((row["Close"] / df["Close"].iloc[-5]) - 1) if len(df) >= 5 else 0.0
        out = {
            "symbol": sym,
            "ok": True,
            "close": round(float(row["Close"]), 2),
            "sma200": round(float(row["SMA200"]), 2),
            "regime": "UP" if regime_up else "DOWN",
            "ret_5d": round(ret5d, 4),
        }
        log.info(f"{C_OK} NIFTY regime: {out['regime']} | 5d={out['ret_5d']}")
        return out
    except Exception as e:
        log.info(f"{C_ERR} Failed regime check: {e}")
        return {"symbol": sym, "ok": False, "reason": str(e)}

# -------- Sentiment (headlines) --------
POS_WORDS = {
    "beats","beat","surge","surges","rises","up","upgrade","record","profit","growth","wins",
    "approval","order","contract","acquires","merger","strong","positive","bullish","expands",
}
NEG_WORDS = {
    "miss","misses","downgrade","plunge","falls","down","loss","losses","decline","fraud",
    "default","probe","raid","ban","delay","weak","negative","bearish","layoff","fire",
}

def headline_sentiment(h: str) -> float:
    h = (h or "").lower()
    pos = sum(1 for w in POS_WORDS if w in h)
    neg = sum(1 for w in NEG_WORDS if w in h)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(1, pos + neg)
    return max(-1.0, min(1.0, score))

def decayed_mean(values: List[Tuple[float, float]]) -> float:
    if not values:
        return 0.0
    num = sum(v * w for v, w in values)
    den = sum(w for _, w in values)
    return num / den if den else 0.0

def get_sentiment_factor(symbol: str) -> float:
    now = pd.Timestamp.now(tz="UTC")
    hl: List[Tuple[float, float]] = []
    half_life = pd.Timedelta(days=3)
    s_upper = normalize_symbol_for_india(symbol).split(".")[0]
    cnt = 0

    for item in NEWS_DB:
        ts = item.get("ts")
        ts = _utc_aware(ts)
        if ts is None or pd.isna(ts):
            continue

        tagged = str(item.get("symbol", "*")).upper()
        headline = str(item.get("headline", ""))
        if tagged not in {"*", s_upper} and s_upper not in headline.upper():
            continue

        age = (now - ts)
        try:
            decay_ratio = float(age / half_life)
        except Exception:
            decay_ratio = 1.0
        weight = 0.5 ** decay_ratio
        score = headline_sentiment(headline)
        hl.append((score, float(weight)))
        cnt += 1

    out = float(max(-1.0, min(1.0, decayed_mean(hl))))
    log.info(f"{C_INFO} Sentiment for {s_upper}: n={cnt}, factor={out:+.2f}")
    return out

def recent_headlines_for(symbol: str, limit: int = 8) -> List[Dict[str, Any]]:
    s_upper = normalize_symbol_for_india(symbol).split(".")[0]
    out: List[Dict[str, Any]] = []
    for it in NEWS_DB[::-1]:  # newest first
        sym = str(it.get("symbol", "*")).upper()
        headline = str(it.get("headline", ""))
        if sym in {"*", s_upper} or s_upper in headline.upper():
            ts = _utc_aware(it.get("ts"))
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            out.append({
                "symbol": sym,
                "headline": headline,
                "sentiment": headline_sentiment(headline),
                "ts": ts_str,
            })
            if len(out) >= limit:
                break
    return out

# -------- Fundamentals (Yahoo info) --------
def safe_get(info: dict, key: str, default=None):
    try:
        v = info.get(key, default)
        if v is None:
            return default
        # many are ratios in 0..1; keep numeric
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        # strings of numbers
        try:
            return float(v)
        except Exception:
            return v
    except Exception:
        return default

def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Fetch a compact fundamentals dict from Yahoo Finance .info. Graceful on failure."""
    if yf is None or not symbol:
        return {"ok": False, "reason": "yfinance unavailable or empty symbol"}
    try:
        ysym = normalize_symbol_for_india(symbol)
        info = yf.Ticker(ysym).info or {}
        # Collate a small subset
        out = {
            "symbol": ysym,
            "trailingPE": safe_get(info, "trailingPE"),
            "forwardPE": safe_get(info, "forwardPE"),
            "pegRatio": safe_get(info, "pegRatio"),
            "returnOnEquity": safe_get(info, "returnOnEquity"),
            "debtToEquity": safe_get(info, "debtToEquity"),
            "profitMargins": safe_get(info, "profitMargins"),
            # growth proxies (may be None depending on source coverage)
            "revenueGrowth": safe_get(info, "revenueGrowth"),
            "earningsQuarterlyGrowth": safe_get(info, "earningsQuarterlyGrowth"),
            "trailingEps": safe_get(info, "trailingEps"),
        }
        out["ok"] = True
        return out
    except Exception as e:
        log.info(f"{C_WARN} Fundamentals fetch failed for {symbol}: {e}")
        return {"ok": False, "reason": str(e)}

def score_fundamentals(fin: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (normalized_score_0_20, breakdown_dict).
    Raw points up to 30, then normalized to 0..20 for blending.
    """
    if not fin or not fin.get("ok"):
        return 0.0, {"ok": False, "reason": fin.get("reason", "no data")}
    pts = 0.0
    br: Dict[str, Any] = {"ok": True, "rules": []}

    # Valuation: PE
    pe_vals = [v for v in [fin.get("trailingPE"), fin.get("forwardPE")] if isinstance(v, (int, float))]
    pe = min(pe_vals) if pe_vals else None
    if pe is not None:
        if pe < 25: pts += 5; br["rules"].append({"factor":"PE","value":pe,"bonus":"+5 (<25)"})
        if pe < 18: pts += 2; br["rules"].append({"factor":"PE","value":pe,"bonus":"+2 (<18)"})
    br["PE_used"] = pe

    # Profitability: ROE
    roe = fin.get("returnOnEquity")
    if isinstance(roe, (int, float)):
        if roe > 0.15: pts += 5; br["rules"].append({"factor":"ROE","value":roe,"bonus":"+5 (>15%)"})
        if roe > 0.20: pts += 2; br["rules"].append({"factor":"ROE","value":roe,"bonus":"+2 (>20%)"})
    br["ROE"] = roe

    # Balance sheet: D/E
    de = fin.get("debtToEquity")
    if isinstance(de, (int, float)):
        if de < 1.0: pts += 5; br["rules"].append({"factor":"D/E","value":de,"bonus":"+5 (<1.0)"})
        if de < 0.5: pts += 2; br["rules"].append({"factor":"D/E","value":de,"bonus":"+2 (<0.5)"})
    br["DebtToEquity"] = de

    # Margins
    pm = fin.get("profitMargins")
    if isinstance(pm, (int, float)):
        if pm > 0.10: pts += 5; br["rules"].append({"factor":"ProfitMargin","value":pm,"bonus":"+5 (>10%)"})
        if pm > 0.15: pts += 2; br["rules"].append({"factor":"ProfitMargin","value":pm,"bonus":"+2 (>15%)"})
    br["ProfitMargins"] = pm

    # Growth (prefer revenueGrowth, fallback EQG)
    gr = fin.get("revenueGrowth")
    if not isinstance(gr, (int, float)):
        gr = fin.get("earningsQuarterlyGrowth")
    if isinstance(gr, (int, float)):
        if gr > 0.05: pts += 5; br["rules"].append({"factor":"Growth","value":gr,"bonus":"+5 (>5%)"})
    br["Growth"] = gr

    # PEG (growth at reasonable price)
    peg = fin.get("pegRatio")
    if isinstance(peg, (int, float)):
        if peg < 1.5: pts += 5; br["rules"].append({"factor":"PEG","value":peg,"bonus":"+5 (<1.5)"})
        if peg < 1.0: pts += 2; br["rules"].append({"factor":"PEG","value":peg,"bonus":"+2 (<1.0)"})
    br["PEG"] = peg

    # Cap raw at 30; normalize to 0..20 for blending
    raw = min(30.0, max(0.0, pts))
    norm = 20.0 * (raw / 30.0)
    br["raw_points_0_30"] = round(raw, 2)
    br["norm_0_20"] = round(norm, 2)
    return float(round(norm, 2)), br

# -------- Scoring & Sizing (technical) --------
def horizon_params(goal: str) -> dict:
    if goal == "short_term":
        return dict(win_rr=1.2, hold_days=5, lookback=50, stop_k=1.0, tgt_k=1.5)
    if goal == "swing":
        return dict(win_rr=1.3, hold_days=15, lookback=100, stop_k=1.2, tgt_k=2.0)
    return dict(win_rr=1.5, hold_days=60, lookback=200, stop_k=1.5, tgt_k=3.0)

def infer_goal_from_market(row: pd.Series) -> str:
    price = float(row["Close"]) or 1.0
    atr_pct = float(row["ATR14"]) / price
    above200 = row["Close"] > row["SMA200"]
    if atr_pct < 0.015 and above200:
        return "long_term"
    if atr_pct < 0.03:
        return "swing"
    return "short_term"

def score_signal(row: pd.Series, goal: str, symbol_for_sentiment: Optional[str] = None) -> tuple[float, dict]:
    comp: Dict[str, float] = {}
    score = 0.0

    # Trend alignment
    trend = 0
    if row["SMA20"] > row["SMA50"]:
        trend += 1
    if row["SMA50"] > row["SMA200"]:
        trend += 1
    comp["trend"] = 20 * trend
    score += comp["trend"]

    # Momentum
    m1 = max(0.0, min(1.0, row["ROC10"] * 10))
    m2 = max(0.0, min(1.0, row["ROC50"] * 10))
    comp["momentum"] = 20 * (0.6 * m1 + 0.4 * m2)
    score += comp["momentum"]

    # Extension vs SMA20 using ATR
    ext = (row["Close"] - row["SMA20"]) / (row["ATR14"] + 1e-9)
    ext_score = 20 * max(0, 1 - abs(ext - 1) / 2)
    comp["extension"] = ext_score
    score += ext_score

    # Volume burst
    comp["volume"] = 10 if int(row.get("VolBurst", 0)) == 1 else 0
    score += comp["volume"]

    # RSI guard
    rsi = row.get("RSI14", np.nan)
    if pd.notna(rsi):
        if 45 <= rsi <= 70:
            comp["rsi"] = 10
        elif rsi < 35:
            comp["rsi"] = -10
        elif rsi > 75:
            comp["rsi"] = -10
        else:
            comp["rsi"] = 0
        score += comp["rsi"]

    # Sentiment
    sent_factor = 0.0
    if symbol_for_sentiment:
        sent_factor = get_sentiment_factor(symbol_for_sentiment)
    comp["sentiment_factor"] = round(float(sent_factor), 2)
    comp["sentiment"] = 10 * float(sent_factor)
    score += comp["sentiment"]

    # Goal bias
    if goal == "long_term" and row["Close"] > row["SMA200"]:
        comp["goal_bias"] = 10
        score += 10
    else:
        comp["goal_bias"] = 0

    log.info(
        f"{C_INFO} TECH Components: trend={comp['trend']:.1f}, mom={comp['momentum']:.1f}, ext={comp['extension']:.1f}, "
        f"vol={comp['volume']:.1f}, rsi={comp.get('rsi',0):.1f}, sent={comp['sentiment']:.1f}, subtotal={score:.1f}"
    )
    return float(round(score, 2)), comp

def risk_fraction(risk_level: str) -> float:
    return {"low": 0.005, "medium": 0.01, "high": 0.02}.get(risk_level, 0.01)

def kelly_cap(prob_win: float, rr: float) -> float:
    edge = prob_win * rr - (1 - prob_win)
    if rr <= 0:
        return 0.0
    return float(max(0.0, min(edge / rr, 0.2)))

# -------- Auto news (RSS) --------
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
DEFAULT_SOURCE_QUERY = (
    "(site:economictimes.indiatimes.com OR site:moneycontrol.com OR site:livemint.com OR site:business-standard.com)"
)

def _url_q(s: str) -> str:
    return s.replace(" ", "%20")

def auto_fetch_headlines_for_symbol(symbol: str, max_items: int = 10) -> int:
    if feedparser is None:
        log.info(f"{C_WARN} feedparser not installed; skipping symbol headlines")
        return 0
    s = normalize_symbol_for_india(symbol).split(".")[0]
    q = f"{s} stock {DEFAULT_SOURCE_QUERY}"
    url = GOOGLE_NEWS_RSS.format(query=_url_q(q))
    try:
        log.info(f"{C_INFO} Fetching headlines for {s} ...")
        feed = feedparser.parse(url)
    except Exception as e:
        log.info(f"{C_ERR} RSS error for {s}: {e}")
        return 0
    count = 0
    for entry in getattr(feed, "entries", [])[:max_items]:
        title = (getattr(entry, "title", None) or "").strip()
        if not title:
            continue
        key = (s.upper(), title.lower())
        if any((it.get("symbol"), str(it.get("headline", "")).lower()) == key for it in NEWS_DB):
            continue
        ts = getattr(entry, "published", None) or getattr(entry, "updated", None)
        try:
            ts_parsed = pd.to_datetime(ts, utc=True) if ts else pd.Timestamp.now(tz="UTC")
        except Exception:
            ts_parsed = pd.Timestamp.now(tz="UTC")

        NEWS_DB.append({"symbol": s.upper(), "headline": title, "ts": ts_parsed})
        count += 1
    log.info(f"{C_OK} Added {count} headlines for {s}")
    return count

def auto_fetch_market_headlines(max_items: int = 20) -> int:
    if feedparser is None:
        log.info(f"{C_WARN} feedparser not installed; skipping market headlines")
        return 0
    q = f"Indian stock market {DEFAULT_SOURCE_QUERY}"
    url = GOOGLE_NEWS_RSS.format(query=_url_q(q))
    try:
        log.info(f"{C_INFO} Fetching market headlines ...")
        feed = feedparser.parse(url)
    except Exception as e:
        log.info(f"{C_ERR} RSS error (market): {e}")
        return 0
    count = 0
    for entry in getattr(feed, "entries", [])[:max_items]:
        title = (getattr(entry, "title", None) or "").strip()
        if not title:
            continue
        key = ("*", title.lower())
        if any((it.get("symbol"), str(it.get("headline", "")).lower()) == key for it in NEWS_DB):
            continue
        ts = getattr(entry, "published", None) or getattr(entry, "updated", None)
        try:
            ts_parsed = pd.to_datetime(ts, utc=True) if ts else pd.Timestamp.now(tz="UTC")
        except Exception:
            ts_parsed = pd.Timestamp.now(tz="UTC")

        NEWS_DB.append({"symbol": "*", "headline": title, "ts": ts_parsed})
        count += 1
    log.info(f"{C_OK} Added {count} market headlines")
    return count


def make_advice(
    df: pd.DataFrame,
    goal: Optional[str],
    wallet: float,
    risk_level: str,
    *,
    symbol_for_sentiment: Optional[str] = None,
    min_confluence: int = 2,
    max_daily_risk_frac: float = 0.02,
) -> AdviceResponse:
    d = compute_features(df)
    log.info(f"{C_INFO} Rows: raw={len(df)} after_features={len(d)}")

    if len(d) < 200:
        log.info(f"{C_ERR} Not enough data after features (need ~200 rows)")
        raise HTTPException(400, "Not enough data. Need at least ~200 rows.")

    row = d.iloc[-1]
    use_goal = goal or infer_goal_from_market(row)
    params = horizon_params(use_goal)

    # Technical score
    tech_score, breakdown = score_signal(row, use_goal, symbol_for_sentiment=symbol_for_sentiment)

    # Fundamentals score (normalized 0..20), blended 20% weight
    fund_info = get_fundamentals(symbol_for_sentiment or "")
    fund_score, fund_break = score_fundamentals(fund_info)

    total_score = round(0.8 * tech_score + 0.2 * fund_score, 2)

    # Score -> win probability (loose calibration)
    prob_win = max(0.35, min(0.7, 0.45 + (total_score - 50) / 350))

    # Entry/stop/targets
    close = float(row["Close"])
    atr = float(row["ATR14"])
    stop = close - params["stop_k"] * atr
    tgt1 = close + params["tgt_k"] * atr
    tgt2 = close + (params["tgt_k"] + 0.5) * atr

    rr = (tgt1 - close) / (close - stop) if (close - stop) > 0 else 0.0

    # Regime guard
    regime = get_nifty_regime()
    regime_ok = bool(regime.get("ok") and regime.get("regime") == "UP")
    if not regime_ok:
        prob_win = max(0.35, prob_win - 0.03)
        log.info(f"{C_WARN} Market regime DOWN — tightening criteria & sizing")

    # Sizing with daily risk cap
    frac1 = risk_fraction(risk_level)
    frac2 = kelly_cap(prob_win, max(1.0, rr))
    base_alloc_frac = max(0.002, min(frac1, frac2 if prob_win > 0.5 else frac1 * 0.7))
    alloc_frac = base_alloc_frac * (1.0 if regime_ok else 0.8)

    max_risk_rupees = wallet * max_daily_risk_frac
    risk_per_share_raw = max(close - stop, 0.01)
    qty_cap_daily = int(max_risk_rupees // risk_per_share_raw)

    max_alloc = wallet * alloc_frac
    risk_per_share = max(risk_per_share_raw, atr * 0.8)
    qty_alloc = int(max_alloc // risk_per_share)
    qty = max(0, min(qty_alloc, qty_cap_daily))

    # Confluence requirement + flags & reasons (still based on technical/sentiment flags)
    flags = {
        "trend_ok": breakdown.get("trend", 0) >= 20,
        "momentum_ok": breakdown.get("momentum", 0) >= 8,
        "volume_ok": breakdown.get("volume", 0) > 0,
        "sentiment_ok": breakdown.get("sentiment", 0) > 0,
    }
    green = sum(1 for v in flags.values() if v)
    min_conf_final = int(min_confluence + (0 if regime_ok else 1))

    reasons: List[str] = []
    if not flags["trend_ok"]:
        reasons.append("Trend not fully aligned (need SMA20>SMA50>SMA200).")
    if not flags["momentum_ok"]:
        reasons.append("Momentum soft (ROC10/50 insufficient).")
    if not flags["volume_ok"]:
        reasons.append("No volume burst (>1.2×MA20).")
    if breakdown.get("rsi", 0) < 0:
        reasons.append("RSI overbought/oversold guard triggered.")
    if breakdown.get("sentiment", 0) <= -5:
        reasons.append("News sentiment adverse.")
    if not regime_ok:
        reasons.append("Market regime DOWN (stricter filters & smaller sizing).")

    decision = "AVOID"
    if total_score >= 65 and qty >= 1 and green >= min_conf_final and breakdown.get("sentiment", 0) > -5:
        decision = "BUY"
    elif 50 <= total_score < 65 and green >= max(1, min_conf_final - 1):
        decision = "HOLD"

    log.info(
        f"{C_INFO} FUND score_norm(0..20)={fund_score:.2f} | TECH={tech_score:.2f} | TOTAL={total_score:.2f}"
    )
    log.info(
        f"{C_OK if decision=='BUY' else (C_WARN if decision=='HOLD' else C_ERR)} "
        f"Decision={decision} | Score={total_score:.1f} | Conf={green}/{min_conf_final} | RR={rr:.2f} | Prob≈{prob_win:.2f}"
    )
    log.info(
        f"{C_INFO} Sizing → Qty={qty}, Entry≈{close:.2f}, Stop={stop:.2f}, T1={tgt1:.2f}, T2={tgt2:.2f}, "
        f"Alloc≈₹{max_alloc:.0f}, Risk/Share≈₹{risk_per_share:.2f}"
    )

    rationale = {
        # keep "score" for UI compatibility + add sub-scores:
        "score": total_score,
        "score_total": total_score,
        "score_tech": tech_score,
        "score_fundamentals": fund_score,
        "components": breakdown,              # technical components
        "fundamentals": fund_break,           # fundamentals breakdown
        "prob_win_est": round(prob_win, 3),
        "rr_est": round(rr, 2),
        "goal": use_goal,
        "atr": round(atr, 4),
        "lookback_used": int(horizon_params(use_goal)["lookback"]),
        "nifty_regime": regime,
        "confluence": int(green),
        "min_confluence": int(min_conf_final),
        "flags": flags,
        "reasons": reasons,
        "sentiment_factor": breakdown.get("sentiment_factor"),
        "headlines_used": recent_headlines_for(symbol_for_sentiment or "", limit=8),
    }

    return AdviceResponse(
        symbol="",  # UI sets this later
        as_of=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        decision=decision,
        rationale=rationale,
        entry_price=round(close, 2),
        stop_loss=round(stop, 2),
        risk_per_share=round(risk_per_share, 2),
        targets=[Target(label="T1", price=round(tgt1, 2)), Target(label="T2", price=round(tgt2, 2))],
        position_qty=max(qty, 0),
        max_alloc_inr=round(max_alloc, 2),
        est_reward_risk=round(rr, 2),
    )

# -------- Routes --------
@app.get("/")
def root():
    return json_ok({
        "ok": True,
        "endpoints": {
            "POST /advice": "Get advice for a symbol",
            "POST /scan": "Rank a list of symbols",
            "POST /news/auto_refresh": "Fetch latest headlines for symbols + market",
            "POST /news/ingest": "Manually add headlines (optional)",
            "POST /news/clear": "Clear stored headlines",
            "POST /news/list": "List headlines + aggregate sentiment",
            "POST /explain": "Explain features/score on a date",
            "GET /dashboard": "Simple UI for scan + news refresh",
            "GET /logs/recent": "Ring-buffer logs",
            "GET /symbols/resolve": "Resolve a human name to a ticker",
        },
        "note": "Pass .NS/.BO for India via Yahoo. Or use data_csv with your own OHLCV file.",
    })

@app.get("/symbols/resolve")
def symbols_resolve(q: str):
    if not q or not q.strip():
        return json_ok({"ok": False, "error": "Empty query"})
    s = q.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO") or s.startswith("^"):
        return json_ok({"ok": True, "input": q, "ticker": s})
    uni = _best_symbol_guess_from_universe(s)
    if uni:
        return json_ok({"ok": True, "input": q, "ticker": f"{uni}.NS"})
    canon = canonicalize_name(s)
    nospace = canon.replace(" ", "")
    return json_ok({"ok": True, "input": q, "ticker": f"{nospace}.NS"})

@app.post("/advice")
def advice(req: AdviceRequest):
    sym_disp = normalize_symbol_for_india(req.symbol)
    log.info("\n" + "-"*70)
    log.info(f"{C_INFO} ADVICE request for {sym_disp} | wallet=₹{req.wallet:.0f} | risk={req.risk_level} | conf≥{req.min_confluence}")
    df = load_ohlcv(req.symbol, req.data_csv)
    auto_fetch_market_headlines(max_items=12)
    auto_fetch_headlines_for_symbol(req.symbol, max_items=8)
    res = make_advice(
        df,
        req.goal,
        req.wallet,
        req.risk_level,
        symbol_for_sentiment=req.symbol,
        min_confluence=req.min_confluence,
        max_daily_risk_frac=req.max_daily_risk_frac,
    )
    res.symbol = sym_disp
    return json_ok(res)

@app.post("/explain")
def explain(req: ExplainRequest):
    df = load_ohlcv(req.symbol, req.data_csv)
    d = compute_features(df)
    if req.on_date:
        dt = pd.to_datetime(req.on_date)
        d = d.loc[:dt]
        if d.empty:
            raise HTTPException(404, "No rows up to that date.")
    row = d.iloc[-1]
    tech_score, parts = score_signal(row, goal="swing", symbol_for_sentiment=req.symbol)
    # fundamentals (optional in explain)
    fin = get_fundamentals(req.symbol)
    fscore, fbreak = score_fundamentals(fin)
    total = round(0.8 * tech_score + 0.2 * fscore, 2)
    payload = {
        "symbol": req.symbol,
        "as_of": str(row.name),
        "close": float(row["Close"]),
        "features": {
            "SMA20": float(row["SMA20"]), "SMA50": float(row["SMA50"]), "SMA200": float(row["SMA200"]),
            "ROC10": float(row["ROC10"]), "ROC50": float(row["ROC50"]), "ATR14": float(row["ATR14"]),
            "RSI14": float(row.get("RSI14")) if pd.notna(row.get("RSI14", np.nan)) else None,
            "VolBurst": int(row.get("VolBurst", 0)),
        },
        "score_tech": tech_score,
        "score_fundamentals": fscore,
        "score_total": total,
        "components": parts,
        "fundamentals": fbreak,
        "sentiment_factor": get_sentiment_factor(req.symbol),
        "nifty_regime": get_nifty_regime(),
    }
    return json_ok(payload)

@app.post("/news/ingest")
def ingest_news(req: IngestNewsRequest):
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    for it in req.items:
        NEWS_DB.append({
            "symbol": (it.symbol or "*").upper(),
            "headline": it.headline.strip(),
            "ts": (_utc_aware(it.ts) if it.ts else now),
        })
    log.info(f"{C_OK} Ingested {len(req.items)} manual headlines; total={len(NEWS_DB)}")
    return json_ok({"ok": True, "count": len(req.items), "total": len(NEWS_DB)})

@app.post("/news/auto_refresh")
def auto_refresh_news(req: AutoNewsRequest):
    total = 0
    total += auto_fetch_market_headlines(max_items=20)
    for s in req.symbols:
        total += auto_fetch_headlines_for_symbol(s, max_items=req.max_per_symbol)
    log.info(f"{C_OK} Auto-refresh headlines added={total}; total={len(NEWS_DB)}")
    return json_ok({"ok": True, "added": total, "total_headlines": len(NEWS_DB)})

@app.post("/news/list")
def news_list(req: AutoNewsRequest):
    symbols = [normalize_symbol_for_india(s).split(".")[0] for s in (req.symbols or [])]
    symbols = list({*(symbols or []), "*"})  # include market by default
    out = []
    for it in NEWS_DB[-500:][::-1]:  # newest-first
        sym = str(it.get("symbol", "*")).upper()
        if sym in symbols:
            h = str(it.get("headline", ""))
            ts = _utc_aware(it.get("ts"))
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            out.append({
                "symbol": sym,
                "headline": h,
                "sentiment": headline_sentiment(h),
                "ts": ts_str,
            })
            if len(out) >= req.max_per_symbol * len(symbols) + 30:
                break
    agg = {s: get_sentiment_factor(s) if s != "*" else get_sentiment_factor("NIFTY") for s in symbols}
    return json_ok({"ok": True, "headlines": out, "aggregate": agg})

@app.post("/news/clear")
def clear_news():
    NEWS_DB.clear()
    log.info(f"{C_OK} Cleared all headlines")
    return json_ok({"ok": True, "total": 0})

@app.post("/scan")
def scan(req: ScanRequest):
    log.info("\n" + "="*70)
    log.info(f"{C_INFO} SCAN {len(req.symbols)} symbols | wallet=₹{req.wallet:.0f} | risk={req.risk_level} | conf≥{req.min_confluence}")
    out: List[Dict[str, Any]] = []
    try:
        auto_refresh_news(AutoNewsRequest(symbols=req.symbols, max_per_symbol=6))
    except Exception as e:
        log.info(f"{C_WARN} News refresh skipped: {e}")

    for s in req.symbols:
        try:
            sym_disp = normalize_symbol_for_india(s)
            log.info("-"*40)
            log.info(f"{C_INFO} Analyzing {sym_disp} ...")
            df = load_ohlcv(s, None)
            res = make_advice(
                df, goal=None, wallet=req.wallet, risk_level=req.risk_level,
                symbol_for_sentiment=s, min_confluence=req.min_confluence,
                max_daily_risk_frac=req.max_daily_risk_frac,
            )
            res.symbol = sym_disp
            out.append(_to_jsonable(res))
        except Exception as e:
            log.info(f"{C_WARN} Skipping {s}: {e}")
            out.append({"symbol": s, "error": str(e)})

    def keyer(x: Dict[str, Any]):
        decision_rank = {"BUY": 0, "HOLD": 1, "AVOID": 2}.get(x.get("decision","AVOID"), 3)
        score = x.get("rationale",{}).get("score", 0)
        try:
            score_val = float(score)
        except Exception:
            score_val = 0.0
        return (decision_rank, -score_val)

    out_sorted = sorted(out, key=keyer)
    log.info(f"{C_OK} Scan complete. BUY candidates: {[o.get('symbol') for o in out_sorted if o.get('decision')=='BUY']}")
    return json_ok({"nifty_regime": get_nifty_regime(), "results": out_sorted})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    regime = get_nifty_regime()
    return templates.TemplateResponse("dashboard.html", {"request": request, "regime": regime})

@app.get("/logs/recent")
def logs_recent(limit: int = 200):
    lines = UI_LOG_HANDLER.buf[-max(10, min(2000, int(limit))):]
    return json_ok({"ok": True, "lines": lines})
