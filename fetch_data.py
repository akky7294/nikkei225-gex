"""
JPX日次相場表PDFを解析してdata/latest.csvに保存するスクリプト
GitHub Actionsから自動実行される

使い方:
  python fetch_data.py --zip Daily_Report_OSE_20260605.zip
  python fetch_data.py --pdf siop_dyr_20260605.pdf
  python fetch_data.py --auto  # inputフォルダの最新ZIPを使う
"""

import argparse
import io
import re
import zipfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# ─── Black-Scholes ───────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (((S / K).__class__(S / K)) and (
        (__import__("math").log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * T**0.5)
    ))
    from scipy.stats import norm as n
    import math
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * T**0.5)
    d2 = d1 - sigma * T**0.5
    if option_type == "call":
        return S * n.cdf(d1) - K * ((2.718281828**(-r * T))) * n.cdf(d2)
    else:
        return K * ((2.718281828**(-r * T))) * n.cdf(-d2) - S * n.cdf(-d1)


def implied_vol(S, K, T, r, market_price, option_type="call", default=0.20):
    import math
    from scipy.stats import norm as n
    if T <= 0 or market_price <= 0:
        return default
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price <= intrinsic * 1.001:
        return default
    try:
        iv = brentq(
            lambda s: bs_price(S, K, T, r, s, option_type) - market_price,
            1e-6, 5.0, xtol=1e-5, maxiter=100
        )
        return iv if 0.01 <= iv <= 5.0 else default
    except Exception:
        return default


# ─── PDF パーサー ─────────────────────────────────────────────────────────────

def parse_pdf(raw: bytes, today: date, spot: float = 38500, r: float = 0.001):
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is required: pip install pdfplumber")

    row_pattern = re.compile(r"(20\d{4})\s+(\d{2}\.\d{2})\s+([\d,]+)\s+\d{6,12}(.*)")
    records = []
    current_type = None

    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "PutOptions" in text:
                current_type = "put"
            if "CallOptions" in text:
                current_type = "call"
            if current_type is None:
                continue

            for line in text.split("\n"):
                m = row_pattern.match(line.strip())
                if not m:
                    continue
                contract_ym = m.group(1)
                exp_md = m.group(2)
                strike_str = m.group(3).replace(",", "")
                rest = m.group(4).strip().split()

                try:
                    strike = int(strike_str)
                except ValueError:
                    continue

                oi = 0
                for token in reversed(rest):
                    clean = token.replace(",", "")
                    if re.fullmatch(r"\d+", clean):
                        oi = int(clean)
                        break
                if oi == 0:
                    continue

                settlement = 0.0
                if len(rest) >= 3:
                    try:
                        v = float(rest[-3].replace(",", ""))
                        if v > 0:
                            settlement = v
                    except ValueError:
                        pass

                year = int(contract_ym[:4])
                try:
                    exp_month = int(exp_md.split(".")[0])
                    exp_day = int(exp_md.split(".")[1])
                    expiry = date(year, exp_month, exp_day)
                except (ValueError, IndexError):
                    continue

                days = (expiry - today).days
                if days <= 0:
                    continue

                T = days / 365.0
                iv = implied_vol(spot, strike, T, r, settlement, current_type) if settlement > 0 else 0.20

                records.append({
                    "date": today.isoformat(),
                    "strike": strike,
                    "expiry": expiry.isoformat(),
                    "type": current_type,
                    "oi": oi,
                    "iv": round(iv, 4),
                    "days_to_expiry": days,
                    "settlement": settlement,
                })

    if not records:
        raise RuntimeError("No data extracted from PDF")

    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} rows (Call:{len(df[df.type=='call'])} / Put:{len(df[df.type=='put'])})")
    return df


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", help="ZIPファイルパス")
    parser.add_argument("--pdf", help="PDFファイルパス")
    parser.add_argument("--auto", action="store_true", help="input/フォルダの最新ZIPを使う")
    parser.add_argument("--spot", type=float, default=38500, help="現値（IV計算用）")
    parser.add_argument("--rate", type=float, default=0.001, help="無リスク金利")
    parser.add_argument("--out", default="data/latest.csv", help="出力CSVパス")
    args = parser.parse_args()

    today = date.today()
    raw_pdf = None

    if args.zip:
        zip_path = Path(args.zip)
        print(f"Reading ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            pdf_files = [n for n in zf.namelist() if "siop" in n.lower() and n.endswith(".pdf")]
            if not pdf_files:
                raise RuntimeError(f"siop_*.pdf not found in {zip_path}")
            print(f"Using: {pdf_files[0]}")
            raw_pdf = zf.read(pdf_files[0])

    elif args.pdf:
        raw_pdf = Path(args.pdf).read_bytes()

    elif args.auto:
        input_dir = Path("input")
        zips = sorted(input_dir.glob("Daily_Report_OSE_*.zip"), reverse=True)
        if not zips:
            print("No ZIP found in input/. Please place Daily_Report_OSE_*.zip there.")
            return
        zip_path = zips[0]
        print(f"Auto-detected: {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            pdf_files = [n for n in zf.namelist() if "siop" in n.lower() and n.endswith(".pdf")]
            raw_pdf = zf.read(pdf_files[0])

    else:
        parser.print_help()
        return

    df = parse_pdf(raw_pdf, today, args.spot, args.rate)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path} ({len(df)} rows)")

    # 日付別アーカイブも保存
    archive_path = Path(f"data/archive/{today.isoformat()}.csv")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(archive_path, index=False)
    print(f"Archived to {archive_path}")


if __name__ == "__main__":
    main()
