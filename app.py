import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================================================================
# CONFIG & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="E-Commerce Analytics & Customer RFM Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2b5c8f;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SYNTHETIC DATA GENERATOR (Fallback for instant demo)
# ==============================================================================
@st.cache_data
def generate_sample_ecommerce_data(n_rows: int = 4000) -> pd.DataFrame:
    """Generates synthetic e-commerce transactional data mimicking Kaggle datasets."""
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    dates = [start_date + timedelta(days=int(np.random.randint(0, days_between_dates))) for _ in range(n_rows)]
    customer_ids = [f"CUST-{np.random.randint(1001, 1400)}" for _ in range(n_rows)]
    order_ids = [f"ORD-{np.random.randint(50000, 52000)}" for _ in range(n_rows)]
    
    categories = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports & Outdoors"]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]
    
    cat_weights = [0.25, 0.30, 0.20, 0.15, 0.10]
    reg_weights = [0.35, 0.30, 0.20, 0.10, 0.05]
    
    chosen_cats = np.random.choice(categories, size=n_rows, p=cat_weights)
    chosen_regs = np.random.choice(regions, size=n_rows, p=reg_weights)
    
    quantities = np.random.choice([1, 2, 3, 4, 5, 8, 10], size=n_rows, p=[0.45, 0.25, 0.12, 0.08, 0.05, 0.03, 0.02])
    unit_prices = np.round(np.random.exponential(scale=45, size=n_rows) + 5, 2)
    
    df = pd.DataFrame({
        "InvoiceNo": order_ids,
        "CustomerID": customer_ids,
        "OrderDate": dates,
        "Category": chosen_cats,
        "Region": chosen_regs,
        "Quantity": quantities,
        "UnitPrice": unit_prices
    })
    
    # Introduce ~2% missing CustomerIDs and 1% negative quantities to simulate raw data
    null_indices = np.random.choice(df.index, size=int(n_rows * 0.02), replace=False)
    df.loc[null_indices, "CustomerID"] = np.nan
    
    return df


# ==============================================================================
# PHASE 1: DATA CLEANING & PREPROCESSING
# ==============================================================================
def clean_ecommerce_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes schema, handles missing values, filters returns/cancellations,
    and calculates line-item revenue.
    """
    df = df_raw.copy()

    # Column name mapping standardizer
    col_mapping = {
        "invoice_no": "InvoiceNo", "invoiceno": "InvoiceNo", "order_id": "InvoiceNo",
        "customer_id": "CustomerID", "customerid": "CustomerID",
        "invoice_date": "OrderDate", "order_date": "OrderDate", "invoicedate": "OrderDate",
        "unit_price": "UnitPrice", "price": "UnitPrice", "unitprice": "UnitPrice", "sales": "UnitPrice",
        "quantity": "Quantity", "qty": "Quantity",
        "region": "Region", "country": "Region",
        "category": "Category", "description": "Category"
    }
    
    # Rename lowercased matching columns
    lower_map = {col: col_mapping[col.lower()] for col in df.columns if col.lower() in col_mapping}
    df = df.rename(columns=lower_map)

    # Validate essential columns
    required_cols = ["InvoiceNo", "OrderDate", "Quantity", "UnitPrice"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: '{col}'. Please check your dataset.")
            return pd.DataFrame()

    # Fallback columns if missing
    if "CustomerID" not in df.columns:
        df["CustomerID"] = "Guest_" + df["InvoiceNo"].astype(str)
    if "Region" not in df.columns:
        df["Region"] = "Global"
    if "Category" not in df.columns:
        df["Category"] = "General Merchandise"

    # Step 1: Drop records where CustomerID is missing (essential for RFM analysis)
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(".0", "", regex=False)

    # Step 2: Datetime Conversion
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df = df.dropna(subset=["OrderDate"])

    # Step 3: Numeric Type Casting & Filtering
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    
    # Filter cancellations (negative quantity) and invalid prices (<= 0)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Step 4: Drop duplicate rows
    df = df.drop_duplicates()

    # Step 5: Derive TotalSales line-item
    df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

    # Step 6: Create Month Period for easy aggregation
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
    
    return df


# ==============================================================================
# PHASE 2: SALES & REVENUE AGGREGATION
# ==============================================================================
def get_monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates monthly revenue and transaction volume."""
    monthly_trend = (
        df.groupby("YearMonth")
        .agg(
            TotalRevenue=("TotalSales", "sum"),
            TotalOrders=("InvoiceNo", "nunique"),
            AverageOrderValue=("TotalSales", "mean")
        )
        .reset_index()
    )
    return monthly_trend

def get_region_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates sales by Region and Product Category."""
    return (
        df.groupby(["Region", "Category"])
        .agg(
            TotalRevenue=("TotalSales", "sum"),
            TotalUnits=("Quantity", "sum")
        )
        .reset_index()
        .sort_values(by="TotalRevenue", ascending=False)
    )


# ==============================================================================
# PHASE 3: CUSTOMER SEGMENTATION (RFM ANALYSIS)
# ==============================================================================
def compute_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary value per customer and assigns
    behavioral tiers via quantile-based scoring.
    """
    # Snapshot reference date: 1 day after the latest transaction date in data
    reference_date = df["OrderDate"].max() + timedelta(days=1)

    # Calculate raw RFM metrics
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("OrderDate", lambda dates: (reference_date - dates.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSales", "sum")
        )
        .reset_index()
    )

    # Quantile binning (1 to 4 scores)
    # Higher Recency = worse (so labels [4, 3, 2, 1] reversed)
    # Higher Frequency & Monetary = better (labels [1, 2, 3, 4])
    try:
        rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    except ValueError:
        rfm["R_Score"] = pd.cut(rfm["Recency"], bins=4, labels=[4, 3, 2, 1]).astype(int)

    try:
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    except ValueError:
        rfm["F_Score"] = pd.cut(rfm["Frequency"], bins=4, labels=[1, 2, 3, 4]).astype(int)

    try:
        rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    except ValueError:
        rfm["M_Score"] = pd.cut(rfm["Monetary"], bins=4, labels=[1, 2, 3, 4]).astype(int)

    # Combined RFM Score and Average Rank
    rfm["RFM_Segment_Code"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["FM_Score"] = (rfm["F_Score"] + rfm["M_Score"]) / 2.0

    # Behavioral Tiers mapping logic
    def assign_tier(row):
        r, fm = row["R_Score"], row["FM_Score"]
        if r >= 3 and fm >= 3.5:
            return "Champions / Loyal"
        elif r >= 3 and fm >= 2:
            return "Potential Loyalists"
        elif r >= 3 and fm < 2:
            return "Recent / New Customers"
        elif r == 2 and fm >= 3:
            return "At Risk (High Value)"
        elif r == 2 and fm < 3:
            return "Need Attention"
        elif r == 1 and fm >= 3:
            return "Can't Lose Them"
        else:
            return "Hibernating / Inactive"

    rfm["Customer_Segment"] = rfm.apply(assign_tier, axis=1)
    return rfm


# ==============================================================================
# PHASE 4: STREAMLIT DASHBOARD INTERFACE
# ==============================================================================
def main():
    st.title("🛒 E-Commerce Sales & Customer Analytics Engine")
    st.caption("Senior Data Scientist Framework: End-to-End ETL, Diagnostic Analytics & RFM Segmentation")

    # Sidebar: Data Source & Filtering
    st.sidebar.header("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload Kaggle CSV / Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Loaded: `{uploaded_file.name}`")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            df_raw = generate_sample_ecommerce_data()
    else:
        st.sidebar.info("Using built-in synthetic benchmark dataset.")
        df_raw = generate_sample_ecommerce_data()

    # Process and Clean Data
    df_clean = clean_ecommerce_data(df_raw)
    if df_clean.empty:
        st.stop()

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    min_date = df_clean["OrderDate"].min().date()
    max_date = df_clean["OrderDate"].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    regions_available = sorted(df_clean["Region"].unique().tolist())
    selected_regions = st.sidebar.multiselect("Regions", options=regions_available, default=regions_available)

    # Apply Filters
    if len(date_range) == 2:
        df_filtered = df_clean[
            (df_clean["OrderDate"].dt.date >= date_range[0]) &
            (df_clean["OrderDate"].dt.date <= date_range[1]) &
            (df_clean["Region"].isin(selected_regions))
        ]
    else:
        df_filtered = df_clean[df_clean["Region"].isin(selected_regions)]

    if df_filtered.empty:
        st.warning("No records found matching the active filter criteria.")
        st.stop()

    # Top KPI Metrics Row
    total_revenue = df_filtered["TotalSales"].sum()
    total_orders = df_filtered["InvoiceNo"].nunique()
    total_customers = df_filtered["CustomerID"].nunique()
    aov = total_revenue / total_orders if total_orders > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("📦 Total Orders", f"{total_orders:,}")
    col3.metric("👥 Unique Customers", f"{total_customers:,}")
    col4.metric("📊 Average Order Value", f"${aov:,.2f}")

    st.markdown("---")

    # Main Navigation Tabs
    tab_trends, tab_category, tab_rfm, tab_data = st.tabs([
        "📈 Sales & Revenue Trends",
        "🏷️ Category & Regional Performance",
        "🎯 RFM Customer Segmentation",
        "📋 Processed Data Explorer"
    ])

    # ---------------- TAB 1: SALES & REVENUE TRENDS ----------------
    with tab_trends:
        st.subheader("Monthly Revenue & Order Volume Dynamics")
        monthly_df = get_monthly_revenue_trend(df_filtered)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_df["YearMonth"].dt.strftime("%Y-%m"),
            y=monthly_df["TotalRevenue"],
            name="Revenue ($)",
            marker_color="#2b5c8f",
            yaxis="y"
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_df["YearMonth"].dt.strftime("%Y-%m"),
            y=monthly_df["TotalOrders"],
            name="Orders Count",
            line=dict(color="#e65100", width=3),
            yaxis="y2"
        ))
        fig_trend.update_layout(
            title="Monthly Sales Revenue vs. Order Count",
            xaxis_title="Month",
            yaxis=dict(title="Revenue ($)", title_font=dict(color="#2b5c8f")),
            yaxis2=dict(title="Order Count", title_font=dict(color="#e65100"), overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ---------------- TAB 2: CATEGORY & REGION ----------------
    with tab_category:
        st.subheader("Top-Selling Categories & Regional Distribution")
        reg_cat_df = get_region_category_sales(df_filtered)

        col_cat_1, col_cat_2 = st.columns(2)
        
        with col_cat_1:
            cat_summary = (
                df_filtered.groupby("Category")["TotalSales"]
                .sum()
                .reset_index()
                .sort_values(by="TotalSales", ascending=True)
            )
            fig_cat = px.bar(
                cat_summary,
                x="TotalSales",
                y="Category",
                orientation="h",
                text_auto="$,.0f",
                title="Revenue by Product Category",
                color="TotalSales",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_cat_2:
            fig_treemap = px.treemap(
                reg_cat_df,
                path=["Region", "Category"],
                values="TotalRevenue",
                title="Revenue Breakdown (Region -> Category)",
                color="TotalRevenue",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_treemap, use_container_width=True)

    # ---------------- TAB 3: RFM SEGMENTATION ----------------
    with tab_rfm:
        st.subheader("Customer Behavioral Cohorts via RFM Analysis")
        rfm_df = compute_rfm_segments(df_filtered)

        col_rfm_1, col_rfm_2 = st.columns([1, 2])

        with col_rfm_1:
            segment_counts = (
                rfm_df["Customer_Segment"]
                .value_counts()
                .reset_index()
            )
            segment_counts.columns = ["Segment", "CustomerCount"]
            
            fig_seg_pie = px.pie(
                segment_counts,
                names="Segment",
                values="CustomerCount",
                hole=0.45,
                title="Customer Distribution by Segment",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_seg_pie, use_container_width=True)

        with col_rfm_2:
            fig_rfm_scatter = px.scatter(
                rfm_df,
                x="Recency",
                y="Monetary",
                size="Frequency",
                color="Customer_Segment",
                hover_name="CustomerID",
                log_y=True,
                title="Recency vs. Monetary (Size = Frequency, Log-Scaled Y)",
                template="plotly_white"
            )
            st.plotly_chart(fig_rfm_scatter, use_container_width=True)

        # Strategic Action Matrix
        st.markdown("### 🎯 Recommended Strategic Actions")
        strategy_col1, strategy_col2, strategy_col3 = st.columns(3)
        with strategy_col1:
            st.success("⭐ **Champions & Loyalists**\n- Exclusive early access\n- VIP loyalty rewards\n- Referral programs")
        with strategy_col2:
            st.warning("⚠️ **At Risk / Can't Lose Them**\n- Win-back discounts\n- Personalized survey email\n- High-touch re-engagement")
        with strategy_col3:
            st.info("🌱 **Recent & Potential Loyalists**\n- Onboarding email drip\n- Cross-sell bundle incentives\n- Review request bonuses")

        # Top Customer Details
        st.markdown("#### Sample Segment Table")
        st.dataframe(
            rfm_df[["CustomerID", "Recency", "Frequency", "Monetary", "Customer_Segment"]]
            .sort_values(by="Monetary", ascending=False)
            .head(50),
            use_container_width=True
        )

    # ---------------- TAB 4: DATA EXPLORER ----------------
    with tab_data:
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df_filtered.head(100), use_container_width=True)
        csv_download = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_download,
            file_name="cleaned_ecommerce_analytics.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
