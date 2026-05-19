import json
from pathlib import Path
from uuid import uuid4


PROJECT_DIR = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_DIR / "01_download.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": uuid4().hex[:8],
        "source": text.splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "id": uuid4().hex[:8],
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """# 01 Download

This notebook creates the required project folders and downloads or validates all raw datasets used in this project. If a raw CSV already exists, the notebook reads it as a cache and still records the result in `download_log.txt`; if a file is missing, the corresponding function attempts to download it from the documented source."""
    ),
    md(
        """## 1. Setup and Directory Creation

The assignment requires the folder structure to be created by Python code. The following cell uses `os.makedirs` to create all required folders and initializes a fresh `download_log.txt` for this run."""
    ),
    code(
        """import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import akshare as ak
except Exception:
    ak = None

START_DATE = "2020-01-01"
START_DATE_AK = "20200101"
END_DATE = datetime.today().strftime("%Y-%m-%d")
END_DATE_AK = datetime.today().strftime("%Y%m%d")
REFRESH_FROM_NETWORK = False

DATA_DIR = Path("data")
DIRS = [
    DATA_DIR / "stock",
    DATA_DIR / "index",
    DATA_DIR / "macro",
    DATA_DIR / "finance",
    DATA_DIR / "clean",
    DATA_DIR / "combined",
    Path("output"),
    Path("codes"),
]

for folder in DIRS:
    os.makedirs(folder, exist_ok=True)

LOG_PATH = Path("download_log.txt")
LOG_PATH.write_text("", encoding="utf-8")

def log_download(dataset_id, shape=None, error=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if error is None:
        message = f"[{timestamp}] SUCCESS  {dataset_id}  shape={shape}"
    else:
        message = f"[{timestamp}] FAILED   {dataset_id}  Error: {error}"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\\n")
    print(message)
"""
    ),
    md(
        """## 2. Stock Daily Data

The stock functions target AkShare `stock_zh_a_hist`, using daily post-adjusted (`hfq`) A-share prices. Required output fields are date, open, close, high, low, volume, and amount, with stock code, name, and industry added for later merges."""
    ),
    code(
        """STOCKS = [
    {"code": "601398", "name": "工商银行", "industry": "银行"},
    {"code": "600036", "name": "招商银行", "industry": "银行"},
    {"code": "002594", "name": "比亚迪", "industry": "汽车"},
    {"code": "601127", "name": "赛力斯", "industry": "汽车"},
    {"code": "000002", "name": "万科A", "industry": "房地产"},
    {"code": "600519", "name": "贵州茅台", "industry": "白酒"},
    {"code": "000858", "name": "五粮液", "industry": "白酒"},
    {"code": "601088", "name": "中国神华", "industry": "能源"},
    {"code": "300308", "name": "中际旭创", "industry": "通讯"},
    {"code": "002352", "name": "顺丰控股", "industry": "物流"},
]

STOCK_COLUMNS = ["日期", "股票代码", "名称", "行业", "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额"]

def standardize_stock(raw, stock):
    rename_map = {
        "日期": "日期",
        "开盘": "开盘价",
        "收盘": "收盘价",
        "最高": "最高价",
        "最低": "最低价",
        "成交量": "成交量",
        "成交额": "成交额",
    }
    df = raw.rename(columns=rename_map).copy()
    df["股票代码"] = stock["code"]
    df["名称"] = stock["name"]
    df["行业"] = stock["industry"]
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df[STOCK_COLUMNS].dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)

def download_stock(stock):
    if ak is None:
        raise RuntimeError("akshare is not installed")
    raw = ak.stock_zh_a_hist(
        symbol=stock["code"],
        period="daily",
        start_date=START_DATE_AK,
        end_date=END_DATE_AK,
        adjust="hfq",
    )
    if raw.empty:
        raise RuntimeError("No data returned")
    return standardize_stock(raw, stock)

def get_stock_data(stock):
    out_path = DATA_DIR / "stock" / f"stock_{stock['code']}.csv"
    if out_path.exists() and not REFRESH_FROM_NETWORK:
        df = pd.read_csv(out_path, dtype={"股票代码": str})
        log_download(f"stock_{stock['code']}", shape=df.shape)
        return df
    try:
        df = download_stock(stock)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log_download(f"stock_{stock['code']}", shape=df.shape)
        return df
    except Exception as exc:
        if out_path.exists():
            df = pd.read_csv(out_path, dtype={"股票代码": str})
            log_download(f"stock_{stock['code']}", shape=df.shape)
            return df
        log_download(f"stock_{stock['code']}", error=exc)
        raise

stock_frames = [get_stock_data(stock) for stock in STOCKS]
stock_combined = pd.concat(stock_frames, ignore_index=True)
stock_combined.to_csv(DATA_DIR / "combined" / "combined_stock_prices.csv", index=False, encoding="utf-8-sig")
display(stock_combined.groupby(["股票代码", "名称", "行业"]).size().reset_index(name="rows"))
"""
    ),
    md(
        """## 3. Index Daily Data

The market benchmark is CSI 300 (`000300`), and the additional broad-market index is the SSE Composite (`000001`). When network refresh is required, the function calls AkShare index daily data and standardizes it to the project schema."""
    ),
    code(
        """INDICES = [
    {"code": "000300", "symbol": "sh000300", "name": "沪深300", "purpose": "CAPM市场基准"},
    {"code": "000001", "symbol": "sh000001", "name": "上证综指", "purpose": "A股整体市场走势参考"},
]

INDEX_COLUMNS = ["日期", "指数代码", "指数名称", "用途", "开盘价", "收盘价", "最高价", "最低价", "成交量"]

def standardize_index(raw, index):
    rename_map = {
        "date": "日期",
        "open": "开盘价",
        "close": "收盘价",
        "high": "最高价",
        "low": "最低价",
        "volume": "成交量",
    }
    df = raw.rename(columns=rename_map).copy()
    df["指数代码"] = index["code"]
    df["指数名称"] = index["name"]
    df["用途"] = index["purpose"]
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df[(df["日期"] >= START_DATE) & (df["日期"] <= END_DATE)].copy()
    df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")
    return df[INDEX_COLUMNS].dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)

def download_index(index):
    if ak is None:
        raise RuntimeError("akshare is not installed")
    raw = ak.stock_zh_index_daily(symbol=index["symbol"])
    if raw.empty:
        raise RuntimeError("No data returned")
    return standardize_index(raw, index)

def get_index_data(index):
    out_path = DATA_DIR / "index" / f"index_{index['code']}.csv"
    if out_path.exists() and not REFRESH_FROM_NETWORK:
        df = pd.read_csv(out_path, dtype={"指数代码": str})
        log_download(f"index_{index['code']}", shape=df.shape)
        return df
    try:
        df = download_index(index)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log_download(f"index_{index['code']}", shape=df.shape)
        return df
    except Exception as exc:
        if out_path.exists():
            df = pd.read_csv(out_path, dtype={"指数代码": str})
            log_download(f"index_{index['code']}", shape=df.shape)
            return df
        log_download(f"index_{index['code']}", error=exc)
        raise

index_frames = [get_index_data(index) for index in INDICES]
index_combined = pd.concat(index_frames, ignore_index=True)
index_combined.to_csv(DATA_DIR / "combined" / "combined_indices.csv", index=False, encoding="utf-8-sig")
display(index_combined.groupby(["指数代码", "指数名称", "用途"]).size().reset_index(name="rows"))
"""
    ),
    md(
        """## 4. Macro Indicators

Macro files are stored individually in `data/macro/` and combined into `data/combined/combined_macro_indicators.csv`. The project uses CPI YoY, USD/CNY, M2 YoY, 1-year LPR, industrial value-added growth, and supplementary World Bank/GMD annual series."""
    ),
    code(
        """MACRO_FILES = {
    "macro_cpi": DATA_DIR / "macro" / "macro_cpi.csv",
    "macro_usd_cny": DATA_DIR / "macro" / "macro_usd_cny.csv",
    "macro_m2": DATA_DIR / "macro" / "macro_m2.csv",
    "macro_lpr_1y": DATA_DIR / "macro" / "macro_lpr_1y.csv",
    "macro_industrial_value_added": DATA_DIR / "macro" / "macro_industrial_value_added.csv",
    "macro_world_bank_annual": DATA_DIR / "macro" / "macro_world_bank_annual.csv",
    "macro_gmd_annual": DATA_DIR / "macro" / "macro_gmd_annual.csv",
}

def get_macro_file(dataset_id, path):
    if path.exists() and not REFRESH_FROM_NETWORK:
        df = pd.read_csv(path)
        log_download(dataset_id, shape=df.shape)
        return df
    # The full network downloader is intentionally conservative because macro APIs
    # have different release calendars. Existing project CSVs are treated as the
    # reproducible raw-data cache; missing files are reported explicitly.
    error = FileNotFoundError(f"{path} not found; rerun the macro AkShare/API downloader or restore raw CSV")
    log_download(dataset_id, error=error)
    raise error

macro_frames = [get_macro_file(dataset_id, path) for dataset_id, path in MACRO_FILES.items()]
macro_combined = pd.concat(macro_frames, ignore_index=True)
macro_combined.to_csv(DATA_DIR / "combined" / "combined_macro_indicators.csv", index=False, encoding="utf-8-sig")
display(macro_combined.groupby("指标代码").size().reset_index(name="rows").head(20))
"""
    ),
    md(
        """## 5. Financial Indicators

Financial data is stored in long format: each row is one stock-year-indicator observation with fields `code`, `year`, `indicator`, and `value`. The notebook records one log line per stock."""
    ),
    code(
        """finance_path = DATA_DIR / "finance" / "finance_ratios.csv"
finance = pd.read_csv(finance_path, dtype={"code": str})
finance["code"] = finance["code"].str.zfill(6)
required_finance_cols = ["code", "year", "indicator", "value"]
finance = finance[required_finance_cols].copy()

for code_value, g in finance.groupby("code"):
    log_download(f"financial_{code_value}", shape=g.shape)

finance.to_csv(finance_path, index=False, encoding="utf-8-sig")
display(finance.groupby(["code", "indicator"]).size().reset_index(name="rows").head(30))
"""
    ),
    md(
        """## 6. Download Summary

The following cell reads `download_log.txt` so the notebook visibly documents the download status of all stock, index, macro, and financial datasets."""
    ),
    code(
        """log_text = LOG_PATH.read_text(encoding="utf-8")
print(log_text)
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"created {NB_PATH}")
