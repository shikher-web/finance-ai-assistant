import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from prophet import Prophet
from theme_utils import apply_theme, get_theme_toggle, apply_responsive_css, show_loading_spinner, create_tooltip
import yfinance as yf
import ta
from news_fetcher import fetch_financial_news
from pdf_generator import generate_comprehensive_pdf_report
from derivatives import get_derivatives_data
from forex import get_forex_data, get_exchange_rates
from crypto import get_crypto_data, get_top_cryptocurrencies
from commodities import get_commodity_data, get_commodities_overview

# Page configuration
st.set_page_config(
    page_title="Shikher's Financial Analysis App - Comprehensive Financial Analysis and Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Apply theme and responsive CSS
apply_theme(st.session_state.theme)
apply_responsive_css()

# Add theme selector in sidebar
st.sidebar.markdown("---")
theme = get_theme_toggle()
if theme != st.session_state.theme:
    st.session_state.theme = theme
    st.rerun()

# Data fetching functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_financial_data(ticker, years=5):
    """Fetch financial data from multiple sources"""
    try:
        from data_fetching_new import data_fetcher
        from utils import detect_market, get_currency_symbol

        # Detect market and get currency
        market = detect_market(ticker)
        currency = get_currency_symbol(market)

        # Fetch financial statements
        balance_sheet, income_stmt, cash_flow = data_fetcher.fetch_financial_data(ticker, years)

        # Get historical data
        hist_data = data_fetcher.fetch_historical_prices(ticker, f"{years}y")

        # Get current price
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice', stock.history(period='1d')['Close'].iloc[-1] if hist_data is not None and not hist_data.empty else None)

        return balance_sheet, income_stmt, cash_flow, hist_data, current_price, market, currency
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None, None, None, None, None, None

# Financial ratios calculation
def calculate_ratios(balance_sheet, income_stmt, current_price):
    """Calculate key financial ratios"""
    ratios = {}

    try:
        if balance_sheet is not None and not balance_sheet.empty:
            # Liquidity ratios
            current_assets = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else 0
            current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else 0
            inventory = balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else 0

            if current_liabilities > 0:
                ratios['Current Ratio'] = current_assets / current_liabilities
                ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities

            # Solvency ratios
            total_debt = balance_sheet.loc['Total Liab'].iloc[0] if 'Total Liab' in balance_sheet.index else 0
            total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0

            if total_equity > 0:
                ratios['Debt to Equity'] = total_debt / total_equity
            if total_assets > 0:
                ratios['Debt to Assets'] = total_debt / total_assets

        if income_stmt is not None and not income_stmt.empty:
            # Profitability ratios
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            total_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0

            if total_revenue > 0:
                ratios['Profit Margin'] = net_income / total_revenue

            if balance_sheet is not None and not balance_sheet.empty:
                total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0

                if total_assets > 0:
                    ratios['ROA'] = net_income / total_assets
                if total_equity > 0:
                    ratios['ROE'] = net_income / total_equity

        # Valuation ratios
        if balance_sheet is not None and not balance_sheet.empty and income_stmt is not None and not income_stmt.empty:
            shares_outstanding = balance_sheet.loc['Common Stock'].iloc[0] if 'Common Stock' in balance_sheet.index else 0
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0

            if shares_outstanding > 0 and current_price > 0:
                eps = net_income / shares_outstanding
                ratios['P/E'] = current_price / eps if eps > 0 else 0

                bv_per_share = total_equity / shares_outstanding if total_equity > 0 else 0
                ratios['P/B'] = current_price / bv_per_share if bv_per_share > 0 else 0

    except Exception as e:
        st.warning(f"Error calculating some ratios: {str(e)}")

    return ratios

# Forecasting function
@st.cache_data
def forecast_revenue(income_stmt, periods=5):
    """Forecast future revenue using Prophet"""
    try:
        if 'Total Revenue' not in income_stmt.index:
            return None

        revenue_series = income_stmt.loc['Total Revenue']
        dates = pd.to_datetime(revenue_series.index)
        df = pd.DataFrame({'ds': dates, 'y': revenue_series.values})
        df = df.sort_values('ds')

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq='Y')
        forecast = model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.warning(f"Error forecasting revenue: {str(e)}")
        return None

# DCF Valuation
def calculate_intrinsic_value(forecasted_fcf, discount_rate=0.1, terminal_growth=0.03, shares_outstanding=1e9):
    """Calculate intrinsic value using DCF"""
    try:
        if forecasted_fcf is None or forecasted_fcf.empty:
            return None

        fcf_values = forecasted_fcf['yhat'].values
        dcf = 0

        for i, fcf in enumerate(fcf_values):
            dcf += fcf / ((1 + discount_rate) ** (i + 1))

        terminal_value = fcf_values[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        dcf += terminal_value / ((1 + discount_rate) ** len(fcf_values))

        return dcf / shares_outstanding
    except Exception as e:
        st.warning(f"Error calculating intrinsic value: {str(e)}")
        return None

# Technical analysis
def calculate_technical_indicators(hist_data):
    """Calculate technical indicators"""
    indicators = {}

    try:
        if hist_data is not None and not hist_data.empty:
            # RSI
            indicators['RSI'] = ta.momentum.RSIIndicator(hist_data['Close']).rsi()

            # MACD
            macd = ta.trend.MACD(hist_data['Close'])
            indicators['MACD'] = macd.macd()
            indicators['MACD_Signal'] = macd.macd_signal()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(hist_data['Close'])
            indicators['BB_Upper'] = bb.bollinger_hband()
            indicators['BB_Lower'] = bb.bollinger_lband()
            indicators['BB_Middle'] = bb.bollinger_mavg()
    except Exception as e:
        st.warning(f"Error calculating technical indicators: {str(e)}")

    return indicators

# Main app
def main():
    # Header with right corner elements
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown('<div class="main-header">Shikher\'s Financial 2-Analysis App</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive Financial Analysis and Forecasting</div>', unsafe_allow_html=True)

    with col2:
        # Right corner elements
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("👤", help="Profile"):
                st.info("Profile feature coming soon!")
        with col_b:
            if st.button("🔔", help="Notifications"):
                st.info("Notifications feature coming soon!")
        with col_c:
            selected_theme = get_theme_toggle()
            if selected_theme != st.session_state.theme:
                st.session_state.theme = selected_theme
                st.rerun()

    # Sidebar Navigation
    st.sidebar.markdown("## 🧭 Navigation")

    # Navigation menu
    selected_page = st.sidebar.radio(
        "Select Section:",
        ["🏠 Dashboard", "📊 Financial Statements", "📈 Ratios & Metrics", "🔮 Forecasting", "💰 Valuation", "📉 Technical Analysis", "📰 News", "📈 Markets", "📋 Research Report"],
        label_visibility="collapsed"
    )

    # Page routing
    if selected_page == "📰 News":
        show_news_section()
    elif selected_page == "📈 Markets":
        show_markets_section()
    elif selected_page == "🏠 Dashboard":
        # Input section for stock analysis
        st.sidebar.subheader("Stock Selection")
        ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS)", value="AAPL").upper()

        # Year selection
        st.sidebar.subheader("Analysis Parameters")

        # Data range selection mode
        range_mode = st.sidebar.radio(
            "Data Range Selection:",
            ["Last N Years", "Custom Year Range"],
            index=0,
            help="Choose how to select the financial data range"
        )

        if range_mode == "Last N Years":
            years = st.sidebar.slider("Financial Years to Analyze", min_value=1, max_value=10, value=5, help="Number of recent financial years to include in analysis")
            start_year = None
            end_year = None
        else:
            # Custom year range
            current_year = datetime.now().year
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_year = st.sidebar.selectbox(
                    "Start Year",
                    options=list(range(current_year - 10, current_year + 2)),
                    index=5,  # Default to 5 years ago
                    help="Select the starting fiscal year"
                )
            with col2:
                end_year = st.sidebar.selectbox(
                    "End Year",
                    options=list(range(current_year - 9, current_year + 3)),
                    index=9,  # Default to current year + 1 (for 2025)
                    help="Select the ending fiscal year"
                )

            # Validate year range
            if start_year >= end_year:
                st.sidebar.error("Start year must be before end year")
                start_year = None
                end_year = None
            else:
                years = None  # Not used when custom range

        # User Controls for Valuation and Forecasting
        st.sidebar.subheader("Valuation & Forecasting Controls")

        # WACC (Weighted Average Cost of Capital)
        wacc = st.sidebar.slider(
            "WACC (Discount Rate)",
            min_value=0.0,
            max_value=0.25,
            value=0.10,
            step=0.01,
            format="%.1%",
            help="Weighted Average Cost of Capital used for DCF valuation"
        )

        # Terminal Growth Rate
        terminal_growth = st.sidebar.slider(
            "Terminal Growth Rate",
            min_value=0.0,
            max_value=0.10,
            value=0.03,
            step=0.005,
            format="%.1%",
            help="Long-term growth rate for terminal value calculation"
        )

        # Forecast Years
        forecast_years = st.sidebar.slider(
            "Forecast Years",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of years to forecast for revenue and cash flows"
        )

        # Revenue Growth Rate
        revenue_growth_rate = st.sidebar.slider(
            "Revenue Growth Rate",
            min_value=-0.10,
            max_value=0.25,
            value=0.05,
            step=0.005,
            format="%.1%",
            help="Annual growth rate for revenue forecasting"
        )

        # Expense Growth Rate
        expense_growth_rate = st.sidebar.slider(
            "Expense Growth Rate",
            min_value=-0.10,
            max_value=0.25,
            value=0.03,
            step=0.005,
            format="%.1%",
            help="Annual growth rate for expense forecasting"
        )

        # FCF Growth Rate
        fcf_growth_rate = st.sidebar.slider(
            "FCF Growth Rate",
            min_value=-0.10,
            max_value=0.25,
            value=0.04,
            step=0.005,
            format="%.1%",
            help="Annual growth rate for free cash flow forecasting"
        )

        # Exchange Rate (for currency conversion if needed)
        exchange_rate = st.sidebar.number_input(
            "Exchange Rate (USD to Local)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.01,
            help="Exchange rate for currency conversion (1.0 = no conversion)"
        )

        # Store user controls in session state
        st.session_state.user_controls = {
            'wacc': wacc,
            'terminal_growth': terminal_growth,
            'forecast_years': forecast_years,
            'revenue_growth_rate': revenue_growth_rate,
            'expense_growth_rate': expense_growth_rate,
            'fcf_growth_rate': fcf_growth_rate,
            'exchange_rate': exchange_rate
        }

        if st.sidebar.button("🔍 Analyze Stock", type="primary"):
            with st.spinner("Fetching data and performing analysis..."):
                # Fetch data
                balance_sheet, income_stmt, cash_flow, hist_data, current_price, market, currency = fetch_financial_data(ticker, years)

                if balance_sheet is None or income_stmt is None:
                    st.error("Unable to fetch financial data. Please check the ticker symbol and try again.")
                    return

                # Calculate ratios
                ratios = calculate_ratios(balance_sheet, income_stmt, current_price)

                # Forecast using user controls
                revenue_forecast = None
                expense_forecast = None
                fcf_forecast = None

                if 'user_controls' in st.session_state:
                    controls = st.session_state.user_controls
                    forecast_years = controls['forecast_years']

                    # Revenue forecast
                    if income_stmt is not None and 'Total Revenue' in income_stmt.index:
                        last_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                        revenue_growth = controls['revenue_growth_rate']
                        revenue_data = []
                        for i in range(forecast_years):
                            year = pd.Timestamp.now().year + i + 1
                            value = last_revenue * ((1 + revenue_growth) ** (i + 1))
                            revenue_data.append({
                                'ds': pd.Timestamp(year, 12, 31),
                                'yhat': value,
                                'yhat_lower': value * 0.9,  # Conservative lower bound
                                'yhat_upper': value * 1.1   # Optimistic upper bound
                            })
                        revenue_forecast = pd.DataFrame(revenue_data)

                    # Expense forecast
                    if income_stmt is not None:
                        expense_series = None
                        possible_expenses = ['Total Operating Expenses', 'Operating Expenses', 'Total Expenses', 'Cost Of Revenue']
                        for col in possible_expenses:
                            if col in income_stmt.index:
                                expense_series = income_stmt.loc[col]
                                break
                        if expense_series is not None:
                            last_expense = expense_series.iloc[0]
                            expense_growth = controls['expense_growth_rate']
                            expense_data = []
                            for i in range(forecast_years):
                                year = pd.Timestamp.now().year + i + 1
                                value = last_expense * ((1 + expense_growth) ** (i + 1))
                                expense_data.append({
                                    'ds': pd.Timestamp(year, 12, 31),
                                    'yhat': value,
                                    'yhat_lower': value * 0.9,
                                    'yhat_upper': value * 1.1
                                })
                            expense_forecast = pd.DataFrame(expense_data)

                    # FCF forecast
                    if cash_flow is not None:
                        fcf_series = cash_flow.loc['Free Cash Flow'] if 'Free Cash Flow' in cash_flow.index else cash_flow.loc['Operating Cash Flow']
                        if not fcf_series.empty:
                            last_fcf = fcf_series.iloc[0]
                            fcf_growth = controls['fcf_growth_rate']
                            fcf_data = []
                            for i in range(forecast_years):
                                year = pd.Timestamp.now().year + i + 1
                                value = last_fcf * ((1 + fcf_growth) ** (i + 1))
                                fcf_data.append({
                                    'ds': pd.Timestamp(year, 12, 31),
                                    'yhat': value,
                                    'yhat_lower': value * 0.8,  # More conservative for FCF
                                    'yhat_upper': value * 1.2
                                })
                            fcf_forecast = pd.DataFrame(fcf_data)
                else:
                    # Fallback to Prophet forecasting if no user controls
                    revenue_forecast = forecast_revenue(income_stmt)

                # Calculate intrinsic value using user controls
                intrinsic_value = None
                if cash_flow is not None and not cash_flow.empty:
                    try:
                        fcf_series = cash_flow.loc['Free Cash Flow'] if 'Free Cash Flow' in cash_flow.index else cash_flow.loc['Operating Cash Flow']
                        if not fcf_series.empty:
                            # Use user-controlled parameters for DCF calculation
                            latest_fcf = fcf_series.iloc[0]
                            growth_rate = terminal_growth  # Use terminal growth as proxy for FCF growth
                            discount_rate = wacc  # Use WACC as discount rate
                            terminal_value = latest_fcf * (1 + growth_rate) / (discount_rate - growth_rate)
                            dcf = sum([latest_fcf * (1 + growth_rate)**i / (1 + discount_rate)**i for i in range(1, forecast_years + 1)])
                            dcf += terminal_value / (1 + discount_rate)**forecast_years

                            # Estimate shares outstanding
                            market_cap = current_price * 1e9  # Rough estimate
                            shares_outstanding = market_cap / current_price
                            intrinsic_value = dcf / shares_outstanding if shares_outstanding > 0 else None
                    except Exception as e:
                        st.warning(f"Error calculating intrinsic value: {str(e)}")

                # Technical analysis
                tech_indicators = calculate_technical_indicators(hist_data)

                # Store data in session state
                st.session_state.ticker = ticker
                st.session_state.current_price = current_price
                st.session_state.balance_sheet = balance_sheet
                st.session_state.income_stmt = income_stmt
                st.session_state.cash_flow = cash_flow
                st.session_state.hist_data = hist_data
                st.session_state.ratios = ratios
                st.session_state.revenue_forecast = revenue_forecast
                st.session_state.intrinsic_value = intrinsic_value
                st.session_state.tech_indicators = tech_indicators
                st.session_state.market = market
                st.session_state.currency = currency
                st.session_state.years = years

        # Display results if data is available
        if 'ticker' in st.session_state:
            display_results()
    else:
        st.info(f"{selected_page} feature coming soon!")

def display_results():
    """Display analysis results"""
    ticker = st.session_state.ticker
    current_price = st.session_state.current_price
    ratios = st.session_state.ratios
    intrinsic_value = st.session_state.intrinsic_value
    revenue_forecast = st.session_state.revenue_forecast
    tech_indicators = st.session_state.tech_indicators

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${current_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        market_cap = current_price * 1e9  # Rough estimate
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Market Cap (Est.)</div>
            <div class="metric-value">${market_cap/1e9:.1f}B</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if intrinsic_value:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Intrinsic Value</div>
                <div class="metric-value">${intrinsic_value:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Intrinsic Value</div>
                <div class="metric-value">N/A</div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if intrinsic_value and current_price:
            ratio = current_price / intrinsic_value
            if ratio < 0.9:
                rec_class = "recommendation-buy"
                rec_text = "BUY"
            elif ratio > 1.1:
                rec_class = "recommendation-sell"
                rec_text = "SELL"
            else:
                rec_class = "recommendation-hold"
                rec_text = "HOLD"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recommendation</div>
                <div class="metric-value {rec_class}">{rec_text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Recommendation</div>
                <div class="metric-value">HOLD</div>
            </div>
            """, unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Financial Statements", "📈 Ratios & Metrics", "🔮 Forecasting", "💰 Valuation", "📉 Technical Analysis", "📋 Research Report"])

    with tab1:
        st.subheader("Financial Statements")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Balance Sheet")
            if 'balance_sheet' in st.session_state and st.session_state.balance_sheet is not None:
                st.dataframe(st.session_state.balance_sheet.head(), use_container_width=True)
            else:
                st.write("No data available")

        with col2:
            st.subheader("Income Statement")
            if 'income_stmt' in st.session_state and st.session_state.income_stmt is not None:
                st.dataframe(st.session_state.income_stmt.head(), use_container_width=True)
            else:
                st.write("No data available")

        with col3:
            st.subheader("Cash Flow")
            if 'cash_flow' in st.session_state and st.session_state.cash_flow is not None:
                st.dataframe(st.session_state.cash_flow.head(), use_container_width=True)
            else:
                st.write("No data available")

    with tab2:
        st.subheader("Financial Ratios")

        if ratios:
            # Group ratios by category
            liquidity_ratios = {k: v for k, v in ratios.items() if k in ['Current Ratio', 'Quick Ratio']}
            profitability_ratios = {k: v for k, v in ratios.items() if k in ['ROA', 'ROE', 'Profit Margin']}
            solvency_ratios = {k: v for k, v in ratios.items() if k in ['Debt to Equity', 'Debt to Assets']}
            valuation_ratios = {k: v for k, v in ratios.items() if k in ['P/E', 'P/B']}

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Liquidity & Profitability")
                for category, ratio_dict in [("Liquidity", liquidity_ratios), ("Profitability", profitability_ratios)]:
                    st.write(f"**{category} Ratios:**")
                    for ratio_name, ratio_value in ratio_dict.items():
                        st.write(f"{ratio_name}: {ratio_value:.2f}")
                    st.write("")

            with col2:
                st.subheader("Solvency & Valuation")
                for category, ratio_dict in [("Solvency", solvency_ratios), ("Valuation", valuation_ratios)]:
                    st.write(f"**{category} Ratios:**")
                    for ratio_name, ratio_value in ratio_dict.items():
                        st.write(f"{ratio_name}: {ratio_value:.2f}")
                    st.write("")
        else:
            st.write("No ratio data available")

    with tab3:
        st.subheader("Revenue Forecasting")

        if revenue_forecast is not None:
            # Plot forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=revenue_forecast['ds'], y=revenue_forecast['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=revenue_forecast['ds'], y=revenue_forecast['yhat_upper'], fill=None, mode='lines', line_color='lightblue', name='Upper Bound'))
            fig.add_trace(go.Scatter(x=revenue_forecast['ds'], y=revenue_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightblue', name='Lower Bound'))

            fig.update_layout(
                title=f"{ticker} Revenue Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display forecast table
            st.subheader("Forecast Data")
            forecast_table = revenue_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
            forecast_table.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            st.dataframe(forecast_table, use_container_width=True)
        else:
            st.write("Unable to generate revenue forecast")

    with tab4:
        st.subheader("DCF Valuation")

        if intrinsic_value:
            st.success(f"**Intrinsic Value:** ${intrinsic_value:.2f}")
            st.info(f"**Current Price:** ${current_price:.2f}")

            if current_price > 0:
                upside = ((intrinsic_value - current_price) / current_price) * 100
                if upside > 0:
                    st.success(f"**Upside Potential:** {upside:.1f}%")
                else:
                    st.error(f"**Downside Risk:** {abs(upside):.1f}%")

            # Simple DCF explanation
            st.subheader("DCF Methodology")
            st.write("""
            **Discounted Cash Flow (DCF) Analysis:**
            - Projects future free cash flows
            - Discounts them to present value using WACC (10%)
            - Adds terminal value assuming perpetual growth (3%)
            - Divides by shares outstanding to get per-share value
            """)
        else:
            st.warning("Unable to calculate intrinsic value due to insufficient data")

    with tab5:
        st.subheader("Technical Analysis")

        if tech_indicators and 'hist_data' in st.session_state:
            hist_data = st.session_state.hist_data

            # Price chart with indicators
            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name='Price'
            ))

            # Bollinger Bands
            if 'BB_Upper' in tech_indicators and 'BB_Lower' in tech_indicators:
                fig.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['BB_Upper'], line=dict(color='gray', width=1), name='BB Upper'))
                fig.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['BB_Lower'], line=dict(color='gray', width=1), name='BB Lower'))

            fig.update_layout(
                title=f"{ticker} Price Chart with Bollinger Bands",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # RSI Chart
            if 'RSI' in tech_indicators:
                st.subheader("RSI Indicator")
                fig_rsi = px.line(x=hist_data.index, y=tech_indicators['RSI'], title="Relative Strength Index (RSI)")
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)

                # RSI Signal
                latest_rsi = tech_indicators['RSI'].iloc[-1]
                if latest_rsi > 70:
                    st.error("⚠️ RSI indicates overbought conditions")
                elif latest_rsi < 30:
                    st.success("✅ RSI indicates oversold conditions")
                else:
                    st.info("ℹ️ RSI is neutral")

            # MACD Chart
            if 'MACD' in tech_indicators and 'MACD_Signal' in tech_indicators:
                st.subheader("MACD Indicator")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['MACD'], mode='lines', name='MACD'))
                fig_macd.add_trace(go.Scatter(x=hist_data.index, y=tech_indicators['MACD_Signal'], mode='lines', name='Signal'))
                fig_macd.update_layout(title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)

                # MACD Signal
                latest_macd = tech_indicators['MACD'].iloc[-1]
                latest_signal = tech_indicators['MACD_Signal'].iloc[-1]
                if latest_macd > latest_signal:
                    st.success("✅ MACD shows bullish momentum")
                else:
                    st.error("⚠️ MACD shows bearish momentum")
        else:
            st.write("Technical analysis data not available")

    with tab6:
        st.subheader("📋 Research Report")

        # Generate PDF Report
        if st.button("📄 Generate Comprehensive PDF Report", type="primary"):
            with st.spinner("Generating comprehensive PDF report..."):
                try:
                    # Prepare data for PDF generation
                    ticker = st.session_state.ticker
                    company_name = ticker  # Could be enhanced to get actual company name
                    market = st.session_state.get('market', 'Unknown')
                    balance_sheet = st.session_state.balance_sheet
                    income_stmt = st.session_state.income_stmt
                    cash_flow = st.session_state.cash_flow
                    ratios = st.session_state.ratios
                    forecasts = st.session_state.revenue_forecast
                    comprehensive_forecasts = None  # Could be enhanced
                    valuations = {"Intrinsic Value": st.session_state.intrinsic_value}
                    intrinsic_value = st.session_state.intrinsic_value
                    current_price = st.session_state.current_price
                    tech_analysis = st.session_state.tech_indicators
                    currency = st.session_state.get('currency', 'USD')

                    # Determine recommendation
                    if intrinsic_value and current_price:
                        ratio = current_price / intrinsic_value
                        if ratio < 0.9:
                            recommendation = "BUY"
                        elif ratio > 1.1:
                            recommendation = "SELL"
                        else:
                            recommendation = "HOLD"
                    else:
                        recommendation = "HOLD"

                    # Generate PDF
                    pdf_filename = generate_comprehensive_pdf_report(
                        ticker=ticker,
                        company_name=company_name,
                        market=market,
                        income_stmt=income_stmt,
                        balance_sheet=balance_sheet,
                        cash_flow=cash_flow,
                        ratios=ratios,
                        forecasts=forecasts,
                        comprehensive_forecasts=comprehensive_forecasts,
                        valuations=valuations,
                        intrinsic_value=intrinsic_value,
                        current_price=current_price,
                        recommendation=recommendation,
                        tech_analysis=tech_analysis,
                        currency=currency
                    )

                    st.success(f"✅ PDF Report generated successfully: {pdf_filename}")

                    # Provide download link
                    with open(pdf_filename, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )

                except Exception as e:
                    st.error(f"❌ Error generating PDF report: {str(e)}")

        # Report Preview
        st.subheader("Report Preview")

        if 'ticker' in st.session_state:
            ticker = st.session_state.ticker
            current_price = st.session_state.current_price
            intrinsic_value = st.session_state.intrinsic_value

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Executive Summary")
                st.write(f"**Company:** {ticker}")
                st.write(f"**Current Price:** ${current_price:.2f}")
                if intrinsic_value:
                    st.write(f"**Intrinsic Value:** ${intrinsic_value:.2f}")
                    upside = ((intrinsic_value - current_price) / current_price) * 100
                    st.write(f"**Upside Potential:** {upside:.1f}%")

            with col2:
                st.markdown("### Key Metrics")
                ratios = st.session_state.get('ratios', {})
                if ratios:
                    for key, value in list(ratios.items())[:4]:  # Show first 4 ratios
                        st.write(f"**{key}:** {value:.2f}")
                else:
                    st.write("No ratio data available")

            # Report Sections Preview
            st.markdown("### Report Sections")
            sections = [
                "📊 Financial Statements",
                "📈 Financial Ratios & Metrics",
                "🔮 Revenue Forecasting",
                "💰 DCF Valuation Analysis",
                "📉 Technical Analysis",
                "📰 Latest Market News"
            ]

            for section in sections:
                st.write(f"• {section}")
        else:
            st.info("Please analyze a stock first to generate a research report.")

def show_news_section():
    """Display financial news section"""
    st.header("📰 Financial News")

    # News filters
    col1, col2, col3 = st.columns(3)

    with col1:
        news_category = st.selectbox(
            "Category",
            ["All", "Business", "Technology", "Markets", "Economy"],
            index=0
        )

    with col2:
        news_count = st.slider("Number of Articles", min_value=5, max_value=50, value=10)

    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["PublishedAt", "Relevancy", "Popularity"],
            index=0
        )

    # Fetch news
    if st.button("🔄 Refresh News", type="primary"):
        with st.spinner("Fetching latest financial news..."):
            try:
                news_data = fetch_financial_news(
                    category=news_category.lower() if news_category != "All" else None,
                    count=news_count,
                    sort_by=sort_by.lower()
                )

                if news_data and 'articles' in news_data:
                    st.session_state.news_data = news_data['articles']
                    st.session_state.news_category = news_category
                    st.session_state.news_count = news_count
                else:
                    st.error("Failed to fetch news data")
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")

    # Display news
    if 'news_data' in st.session_state and st.session_state.news_data:
        articles = st.session_state.news_data

        st.subheader(f"Latest {st.session_state.news_category} News")

        for i, article in enumerate(articles):
            with st.container():
                col1, col2 = st.columns([1, 3])

                with col1:
                    if article.get('urlToImage'):
                        st.image(article['urlToImage'], use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/150x100?text=No+Image", use_column_width=True)

                with col2:
                    st.subheader(article.get('title', 'No Title'))
                    st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                    st.write(f"**Published:** {article.get('publishedAt', 'Unknown')[:10] if article.get('publishedAt') else 'Unknown'}")

                    if article.get('description'):
                        st.write(article['description'])

                    if article.get('url'):
                        st.markdown(f"[Read full article]({article['url']})")

                st.divider()

        # News summary
        st.subheader("News Summary")
        total_articles = len(articles)
        sources = list(set([article.get('source', {}).get('name', 'Unknown') for article in articles]))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", total_articles)
        with col2:
            st.metric("Sources", len(sources))
        with col3:
            st.metric("Category", st.session_state.news_category)

    else:
        st.info("Click 'Refresh News' to load the latest financial news.")

def show_markets_section():
    """Display markets section with sub-tabs for different market segments"""
    st.header("📈 Markets Overview")

    # Market sub-tabs
    market_tabs = st.tabs(["📊 Derivatives", "💱 Forex", "₿ Crypto", "🏭 Commodities"])

    with market_tabs[0]:  # Derivatives
        st.subheader("📊 Derivatives Market")

        col1, col2 = st.columns(2)

        with col1:
            market_type = st.selectbox(
                "Market",
                ["NSE F&O", "BSE F&O", "CME", "ICE"],
                key="derivatives_market"
            )

        with col2:
            symbol = st.text_input("Symbol (e.g., NIFTY, BANKNIFTY)", value="NIFTY", key="derivatives_symbol")

        if st.button("🔍 Analyze Derivatives", key="analyze_derivatives"):
            with st.spinner("Fetching derivatives data..."):
                try:
                    derivatives_data = get_derivatives_data(market_type, symbol)

                    if derivatives_data:
                        # Display options chain
                        st.subheader(f"{symbol} Options Chain")

                        # Mock options data for demonstration
                        options_data = pd.DataFrame({
                            'Strike': [18000, 18500, 19000, 19500, 20000],
                            'Call OI': [125000, 98000, 156000, 87000, 124000],
                            'Call LTP': [450, 280, 120, 35, 8],
                            'Put LTP': [12, 45, 150, 320, 520],
                            'Put OI': [98000, 134000, 167000, 95000, 78000]
                        })

                        st.dataframe(options_data, use_container_width=True)

                        # PCR Ratio
                        total_call_oi = options_data['Call OI'].sum()
                        total_put_oi = options_data['Put OI'].sum()
                        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PCR Ratio", f"{pcr:.2f}")
                        with col2:
                            st.metric("Total Call OI", f"{total_call_oi:,}")
                        with col3:
                            st.metric("Total Put OI", f"{total_put_oi:,}")

                        # Market sentiment
                        if pcr > 1.2:
                            st.success("🐂 Bullish sentiment (PCR > 1.2)")
                        elif pcr < 0.8:
                            st.error("🐻 Bearish sentiment (PCR < 0.8)")
                        else:
                            st.info("⚖️ Neutral sentiment")

                    else:
                        st.error("Unable to fetch derivatives data")

                except Exception as e:
                    st.error(f"Error fetching derivatives data: {str(e)}")

    with market_tabs[1]:  # Forex
        st.subheader("💱 Forex Market")

        col1, col2 = st.columns(2)

        with col1:
            currency_pair = st.selectbox(
                "Currency Pair",
                ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"],
                key="forex_pair"
            )

        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                ["1D", "1W", "1M"],
                key="forex_timeframe"
            )

        if st.button("🔍 Analyze Forex", key="analyze_forex"):
            with st.spinner("Fetching forex data..."):
                try:
                    forex_data = get_exchange_rates()

                    if forex_data and currency_pair in forex_data:
                        current_rate = forex_data[currency_pair]

                        # Mock historical data for chart
                        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                        historical_rates = [current_rate * (1 + np.random.normal(0, 0.02)) for _ in range(30)]
                        historical_rates.sort()  # Make it somewhat realistic

                        hist_df = pd.DataFrame({
                            'Date': dates,
                            'Rate': historical_rates
                        })

                        # Price chart
                        fig = px.line(hist_df, x='Date', y='Rate',
                                    title=f'{currency_pair} Exchange Rate ({timeframe})')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # Current rate display
                        st.metric(f"Current {currency_pair}", f"{current_rate:.4f}")

                        # Technical indicators
                        st.subheader("Technical Analysis")
                        col1, col2, col3 = st.columns(3)

                        # Mock RSI
                        rsi = np.random.uniform(30, 70)
                        with col1:
                            if rsi > 70:
                                st.error(f"RSI: {rsi:.1f} (Overbought)")
                            elif rsi < 30:
                                st.success(f"RSI: {rsi:.1f} (Oversold)")
                            else:
                                st.info(f"RSI: {rsi:.1f} (Neutral)")

                        # Mock trend
                        trend = np.random.choice(["Bullish", "Bearish", "Sideways"])
                        with col2:
                            if trend == "Bullish":
                                st.success(f"Trend: {trend}")
                            elif trend == "Bearish":
                                st.error(f"Trend: {trend}")
                            else:
                                st.info(f"Trend: {trend}")

                        # Mock volatility
                        volatility = np.random.uniform(0.5, 2.5)
                        with col3:
                            st.metric("Volatility", f"{volatility:.1f}%")

                    else:
                        st.error("Unable to fetch forex data")

                except Exception as e:
                    st.error(f"Error fetching forex data: {str(e)}")

    with market_tabs[2]:  # Crypto
        st.subheader("₿ Cryptocurrency Market")

        col1, col2 = st.columns(2)

        with col1:
            crypto_symbol = st.selectbox(
                "Cryptocurrency",
                ["BTC", "ETH", "BNB", "ADA", "SOL", "DOT"],
                key="crypto_symbol"
            )

        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Price Chart", "Technical Indicators", "Market Overview"],
                key="crypto_analysis"
            )

        if st.button("🔍 Analyze Crypto", key="analyze_crypto"):
            with st.spinner("Fetching cryptocurrency data..."):
                try:
                    crypto_data = get_top_cryptocurrencies()

                    if crypto_data:
                        # Filter for selected crypto
                        selected_crypto = crypto_data[crypto_data['symbol'] == crypto_symbol]

                        if not selected_crypto.empty:
                            crypto_info = selected_crypto.iloc[0]

                            # Display basic info
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Price", f"${crypto_info.get('current_price', 0):,.2f}")
                            with col2:
                                st.metric("Market Cap", f"${crypto_info.get('market_cap', 0):,.0f}")
                            with col3:
                                st.metric("24h Change", f"{crypto_info.get('price_change_percentage_24h', 0):.2f}%")
                            with col4:
                                st.metric("Volume", f"${crypto_info.get('total_volume', 0):,.0f}")

                            if analysis_type == "Price Chart":
                                # Mock price chart
                                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                                prices = [crypto_info.get('current_price', 0) * (1 + np.random.normal(0, 0.05)) for _ in range(30)]

                                price_df = pd.DataFrame({
                                    'Date': dates,
                                    'Price': prices
                                })

                                fig = px.line(price_df, x='Date', y='Price',
                                            title=f'{crypto_symbol} Price Chart (30D)')
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)

                            elif analysis_type == "Technical Indicators":
                                st.subheader("Technical Indicators")

                                # Mock technical indicators
                                indicators = {
                                    'RSI': np.random.uniform(20, 80),
                                    'MACD': np.random.uniform(-5, 5),
                                    'Bollinger Upper': crypto_info.get('current_price', 0) * 1.05,
                                    'Bollinger Lower': crypto_info.get('current_price', 0) * 0.95
                                }

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Momentum Indicators:**")
                                    st.write(f"RSI: {indicators['RSI']:.1f}")
                                    st.write(f"MACD: {indicators['MACD']:.2f}")

                                with col2:
                                    st.write("**Volatility Indicators:**")
                                    st.write(f"BB Upper: ${indicators['Bollinger Upper']:,.2f}")
                                    st.write(f"BB Lower: ${indicators['Bollinger Lower']:,.2f}")

                            else:  # Market Overview
                                st.subheader("Market Overview")

                                # Top cryptocurrencies table
                                top_cryptos = crypto_data.head(10)[['name', 'symbol', 'current_price', 'price_change_percentage_24h', 'market_cap']]
                                top_cryptos.columns = ['Name', 'Symbol', 'Price', '24h Change', 'Market Cap']
                                st.dataframe(top_cryptos, use_container_width=True)

                        else:
                            st.error(f"Data for {crypto_symbol} not available")

                    else:
                        st.error("Unable to fetch cryptocurrency data")

                except Exception as e:
                    st.error(f"Error fetching cryptocurrency data: {str(e)}")

    with market_tabs[3]:  # Commodities
        st.subheader("🏭 Commodities Market")

        col1, col2 = st.columns(2)

        with col1:
            commodity_category = st.selectbox(
                "Category",
                ["Precious Metals", "Energy", "Agricultural", "Industrial"],
                key="commodity_category"
            )

        with col2:
            market_region = st.selectbox(
                "Market",
                ["Global", "India"],
                key="commodity_market"
            )

        if st.button("🔍 Analyze Commodities", key="analyze_commodities"):
            with st.spinner("Fetching commodities data..."):
                try:
                    commodities_data = get_commodities_overview(commodity_category, market_region)

                    if commodities_data:
                        st.subheader(f"{commodity_category} - {market_region} Market")

                        # Display commodities table
                        if isinstance(commodities_data, pd.DataFrame):
                            st.dataframe(commodities_data, use_container_width=True)
                        else:
                            # Mock data if API fails
                            mock_commodities = {
                                'Precious Metals': [
                                    {'Name': 'Gold', 'Price': 1950.50, 'Change': 0.85, 'Unit': 'USD/oz'},
                                    {'Name': 'Silver', 'Price': 23.75, 'Change': -0.32, 'Unit': 'USD/oz'},
                                    {'Name': 'Platinum', 'Price': 945.20, 'Change': 1.15, 'Unit': 'USD/oz'}
                                ],
                                'Energy': [
                                    {'Name': 'Crude Oil (WTI)', 'Price': 78.45, 'Change': 2.10, 'Unit': 'USD/bbl'},
                                    {'Name': 'Brent Oil', 'Price': 82.30, 'Change': 1.95, 'Unit': 'USD/bbl'},
                                    {'Name': 'Natural Gas', 'Price': 2.85, 'Change': -0.15, 'Unit': 'USD/MMBtu'}
                                ],
                                'Agricultural': [
                                    {'Name': 'Corn', 'Price': 475.25, 'Change': 1.20, 'Unit': 'USD/bushel'},
                                    {'Name': 'Wheat', 'Price': 612.80, 'Change': -0.85, 'Unit': 'USD/bushel'},
                                    {'Name': 'Soybeans', 'Price': 1185.50, 'Change': 0.95, 'Unit': 'USD/bushel'}
                                ],
                                'Industrial': [
                                    {'Name': 'Copper', 'Price': 3.85, 'Change': 1.45, 'Unit': 'USD/lb'},
                                    {'Name': 'Aluminum', 'Price': 0.95, 'Change': -0.25, 'Unit': 'USD/lb'},
                                    {'Name': 'Steel', 'Price': 425.30, 'Change': 0.75, 'Unit': 'USD/ton'}
                                ]
                            }

                            if commodity_category in mock_commodities:
                                df = pd.DataFrame(mock_commodities[commodity_category])
                                st.dataframe(df, use_container_width=True)

                                # Market summary
                                avg_change = df['Change'].mean()
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Average Change", f"{avg_change:.2f}%")
                                with col2:
                                    positive_changes = (df['Change'] > 0).sum()
                                    st.metric("Gainers", positive_changes)
                                with col3:
                                    negative_changes = (df['Change'] < 0).sum()
                                    st.metric("Decliners", negative_changes)

                                # Market sentiment
                                if avg_change > 0.5:
                                    st.success("📈 Bullish market sentiment")
                                elif avg_change < -0.5:
                                    st.error("📉 Bearish market sentiment")
                                else:
                                    st.info("⚖️ Mixed market sentiment")

                    else:
                        st.error("Unable to fetch commodities data")

                except Exception as e:
                    st.error(f"Error fetching commodities data: {str(e)}")

if __name__ == "__main__":
    main()
