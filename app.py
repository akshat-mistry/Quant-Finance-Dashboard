import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt

# ================== FORMATTERS ==================
def fmt_currency(x):
    if x is None:
        return "N/A"
    x = float(x)
    if x >= 1e12:
        return f"₹{x/1e12:.2f}T"
    elif x >= 1e9:
        return f"₹{x/1e9:.2f}B"
    elif x >= 1e7:
        return f"₹{x/1e7:.2f}Cr"
    elif x >= 1e5:
        return f"₹{x/1e5:.2f}L"
    else:
        return f"₹{x:,.2f}"

def fmt_price(x):
    return "N/A" if x is None else f"₹{x:,.2f}"

def fmt_volume(x):
    if x is None:
        return "N/A"
    x = float(x)
    if x >= 1e7:
        return f"{x/1e7:.2f}Cr"
    elif x >= 1e6:
        return f"{x/1e6:.2f}M"
    elif x >= 1e5:
        return f"{x/1e5:.2f}L"
    else:
        return f"{int(x):,}"

def fmt_count(x):
    if x is None:
        return "N/A"
    x = float(x)
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    elif x >= 1e3:
        return f"{x/1e3:.2f}K"
    else:
        return f"{int(x)}"

def fmt_epoch_date(epoch):
    if epoch is None:
        return "N/A"
    try:
        return dt.fromtimestamp(epoch).strftime("%b %d, %Y")
    except:
        return "N/A"

# ================== APP CONFIG ==================
st.set_page_config(page_title="Quant Stock Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .topbar {
        background-color: #0f172a;
        color: white;
        padding: 14px 20px;
        font-size: 22px;
        font-weight: 600;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="topbar">📈 Quant Stock Dashboard</div>', unsafe_allow_html=True)

st.sidebar.header("Menu")
menu = st.sidebar.radio(
    "Select Option",
    ["Check Stock Analysis", "Compare Two Stocks", "Portfolio Making"]
)

# ================== STOCK DATABASE ==================
stocks_db = {
    "RELIANCE": "Reliance Industries",        # NOT RIL
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "AXISBANK": "Axis Bank",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "ITC": "ITC Limited",
    "HINDUNILVR": "Hindustan Unilever",
    "BHARTIARTL": "Bharti Airtel",
    "ASIANPAINT": "Asian Paints",
    "MARUTI": "Maruti Suzuki",
    "HCLTECH": "HCL Technologies",
    "WIPRO": "Wipro",
    "SUNPHARMA": "Sun Pharmaceutical",
    "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",
    "ULTRACEMCO": "UltraTech Cement",
    "TITAN": "Titan Company",
    "ONGC": "Oil & Natural Gas Corp",
    "NTPC": "NTPC Limited",
    "POWERGRID": "Power Grid Corp",
    "COALINDIA": "Coal India",
    "JSWSTEEL": "JSW Steel",
    "TATASTEEL": "Tata Steel",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "GRASIM": "Grasim Industries",
    "DIVISLAB": "Divi's Laboratories",
    "DRREDDY": "Dr Reddy's Laboratories",
    "EICHERMOT": "Eicher Motors",
    "HEROMOTOCO": "Hero MotoCorp",
    "HDFCLIFE": "HDFC Life Insurance",
    "SBILIFE": "SBI Life Insurance",
    "INDUSINDBK": "IndusInd Bank",
    "BRITANNIA": "Britannia Industries",
    "CIPLA": "Cipla",
    "TECHM": "Tech Mahindra",
    "APOLLOHOSP": "Apollo Hospitals",
    "TATAMOTORS": "Tata Motors",
    "UPL": "UPL Limited",
    "NESTLEIND": "Nestle India",
    "HAVELLS": "Havells India",
    "PIDILITIND": "Pidilite Industries",
    "SHREECEM": "Shree Cement",
    "BPCL": "Bharat Petroleum",
    "IOC": "Indian Oil Corporation"
}


# ================== MAIN ==================
if menu == "Check Stock Analysis":

    st.markdown("### Time Duration")
    duration = st.radio(
        "", ["1D","5D","1M","6M","YTD","1Y","5Y","MAX","Custom"], horizontal=True
    )

    period, start_date, end_date = "max", None, None
    if duration == "1D": period = "1d"
    elif duration == "5D": period = "5d"
    elif duration == "1M": period = "1mo"
    elif duration == "6M": period = "6mo"
    elif duration == "1Y": period = "1y"
    elif duration == "5Y": period = "5y"
    elif duration == "YTD":
        start_date = datetime.date(datetime.date.today().year, 1, 1)
        end_date = datetime.date.today()
    elif duration == "Custom":
        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input("Start Date")
        with c2: end_date = st.date_input("End Date")

    st.subheader("Select Stock")
    q = st.text_input("Search stocks (name or symbol)")

    df = pd.DataFrame([{"Symbol":k,"Company":v} for k,v in stocks_db.items()])
    if q:
        q = q.lower()
        df = df[df["Symbol"].str.lower().str.contains(q) |
                df["Company"].str.lower().str.contains(q)]

    st.dataframe(df, hide_index=True, use_container_width=True)
    
    symbol = st.selectbox(
        "Select stock",
        df["Symbol"].tolist()
    )


    if symbol:
        ticker = yf.Ticker(symbol + ".NS")
        try:
            data = (ticker.history(start=start_date, end=end_date)
                    if duration in ["YTD","Custom"]
                    else ticker.history(period=period))
            if data.empty or len(data) < 5:
                st.warning("Selected range unavailable. Showing full history.")
                data = ticker.history(period="max")
        except:
            st.error("Error fetching data")
            st.stop()

        info = ticker.info
        st.success(f"{stocks_db[symbol]} ({symbol}.NS)")

        # ================== OVERVIEW ==================
        st.subheader("Overview")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("Open", fmt_price(info.get("open")))
            st.write("High", fmt_price(info.get("dayHigh")))
            st.write("Low", fmt_price(info.get("dayLow")))
            st.write("Market Cap", fmt_currency(info.get("marketCap")))
            st.write("Avg. Volume", fmt_volume(info.get("averageVolume")))

        with c2:
            period_volume = data["Volume"].sum()
            st.write("Volume (Selected Period)", fmt_volume(period_volume))
            st.write("Dividend Yield", info.get("dividendYield","N/A"))
            st.write("Quarterly Dividend", fmt_price(info.get("trailingAnnualDividendRate")))
            st.write("Ex-Dividend Date", fmt_epoch_date(info.get("exDividendDate")))
            st.write("P/E Ratio", info.get("trailingPE","N/A"))

        with c3:
            st.write("52W High", fmt_price(info.get("fiftyTwoWeekHigh")))
            st.write("52W Low", fmt_price(info.get("fiftyTwoWeekLow")))
            st.write("EPS", fmt_price(info.get("trailingEps")))
            st.write("Shares Outstanding", fmt_count(info.get("sharesOutstanding")))
            st.write("Employees", fmt_count(info.get("fullTimeEmployees")))

        # ================== CHART ==================
        st.subheader("Price Chart")
        st.line_chart(data["Close"])

        # ================== QUANT ==================
        st.subheader("Quantitative Analysis")
        data["Returns"] = data["Close"].pct_change()

        daily_vol = data["Returns"].std()
        annual_vol = daily_vol * np.sqrt(252)
        risk = "Low 🟢" if annual_vol < 0.20 else "Medium 🟡" if annual_vol < 0.35 else "High 🔴"

        # ---- Market Sentiment (Safe) ----
        window = min(20, len(data) // 2)

        if window > 0:
            trend = data["Close"].iloc[-1] - data["Close"].iloc[-window]
            sentiment = "Bullish 📈" if trend > 0 else "Bearish 📉"
        else:
            sentiment = "Neutral ⚪"


        q1,q2,q3 = st.columns(3)
        q1.metric("Daily Volatility", f"{daily_vol:.4f}")
        q2.metric("Annualized Volatility", f"{annual_vol:.2%}")
        q3.metric("Risk Level", risk)

        st.write("Market Sentiment:", sentiment)

        # ================== PRICE RANGE ==================
        st.subheader("Price Range Analysis")
        start_p = data["Close"].iloc[0]
        hi, lo = data["High"].max(), data["Low"].min()
        avg_price = data["Close"].mean()
        range_vol = ((hi - lo) / avg_price) * 100


        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Start Price", fmt_price(start_p))
        r2.metric("Period High", fmt_price(hi))
        r3.metric("Period Low", fmt_price(lo))
        r4.metric("Range Volatility (Avg - Normalized)", f"{range_vol:.2f}%")

elif menu == "Compare Two Stocks":

    st.subheader("📊 Stock Comparison")

    # -------- Select Stocks --------
    compare_symbols = st.multiselect(
        "Select stocks to compare (min 2, max 5)",
        options=list(stocks_db.keys()),
        max_selections=5
    )

    if len(compare_symbols) < 2:
        st.info("Please select at least 2 stocks to compare.")
        st.stop()

    # -------- Time Duration --------
    st.markdown("### Time Duration")

    duration = st.radio(
        "",
        ["1M", "6M", "1Y", "5Y", "MAX"],
        horizontal=True,
        key="compare_duration"
    )

    period_map = {
        "1M": "1mo",
        "6M": "6mo",
        "1Y": "1y",
        "5Y": "5y",
        "MAX": "max"
    }

    period = period_map[duration]

    comparison_rows = []
    price_df = pd.DataFrame()
    start_dates = {}

    # -------- Fetch + Compute Metrics --------
    for symbol in compare_symbols:

        ticker = yf.Ticker(symbol + ".NS")
        data = ticker.history(period=period)

        if data.empty or len(data) < 5:
            continue

        # Save listing start date
        start_dates[symbol] = data.index.min()

        # Price metrics
        start_price = data["Close"].iloc[0]
        end_price = data["Close"].iloc[-1]
        period_return = ((end_price - start_price) / start_price) * 100

        # Volatility
        data["Returns"] = data["Close"].pct_change()
        daily_vol = data["Returns"].std()
        annual_vol = daily_vol * np.sqrt(252)

        # Risk level
        if annual_vol < 0.20:
            risk = "Low 🟢"
        elif annual_vol < 0.35:
            risk = "Medium 🟡"
        else:
            risk = "High 🔴"

        # Price range volatility (avg normalized)
        hi = data["High"].max()
        lo = data["Low"].min()
        avg_price = data["Close"].mean()
        range_vol = ((hi - lo) / avg_price) * 100

        # Store row
        comparison_rows.append({
            "Stock": symbol,
            "Period Return (%)": round(period_return, 2),
            "Annual Volatility (%)": round(annual_vol * 100, 2),
            "Range Volatility (%)": round(range_vol, 2),
            "Risk Level": risk
        })

        # Price data for charts
        price_df[symbol] = data["Close"]

    # -------- Align all stocks to common overlapping date --------
    if not start_dates:
        st.error("Not enough data to compare selected stocks.")
        st.stop()

    common_start = max(start_dates.values())
    price_df = price_df[price_df.index >= common_start]

    if len(price_df) < 5:
        st.warning(
            "⚠️ Selected stocks do not have sufficient overlapping history "
            "for the chosen time range."
        )
        st.stop()

    # -------- Comparison Table --------
    st.subheader("📋 Quantitative Comparison Table")

    comp_df = pd.DataFrame(comparison_rows)

    if comp_df.empty:
        st.error("Not enough data to compare selected stocks.")
        st.stop()

    st.dataframe(comp_df, use_container_width=True)

    # -------- Price Comparison Charts --------
    st.subheader("📈 Price Comparison Charts")

    # Normalized Performance
    st.markdown("#### Normalized Performance (Base = 100)")
    norm_price_df = price_df / price_df.iloc[0] * 100
    st.line_chart(norm_price_df)

    # Actual Prices
    st.markdown("#### Actual Price Movement (₹)")
    st.line_chart(price_df)

    # -------- Interpretation Help --------
    st.markdown("### 🧠 How to Read This")
    st.markdown("""
    - **Period Return (%)**: Total return over selected duration  
    - **Annual Volatility**: Risk derived from daily price fluctuations  
    - **Range Volatility**: High–Low movement normalized by average price  
    - **Risk Level**: Qualitative interpretation of volatility  
    - **Normalized Chart**: Relative performance comparison  
    - **Actual Price Chart**: Absolute price levels  
    """)


elif menu == "Portfolio Making":
    st.subheader("Portfolio Making")
    st.write("Coming soon…")
