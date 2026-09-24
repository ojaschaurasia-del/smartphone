
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smartphone Intelligence Lab",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. SIMPLE PROFESSIONAL DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 32px;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827 0%, #172554 55%, #312e81 100%);
        border: 1px solid #334155;
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 17px;
        margin-top: 8px;
    }

    .info-box {
        padding: 16px 20px;
        border-radius: 14px;
        background: #172554;
        border: 1px solid #334155;
        color: #dbeafe;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "raw_data": None,
    "clean_data": None,
    "cleaning_report": {},
    "file_signature": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 4. DATA-CLEANING / FEATURE-EXTRACTION HELPERS
# ============================================================

MISSING_MARKERS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
    "--",
    "not available",
    "not_applicable",
    "not applicable",
}


def standardize_column_names(df):
    df = df.copy()
    names = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    result = []
    counts = {}
    for name in names:
        base = name if name else "unnamed_column"
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")

    df.columns = result
    return df


def standardize_missing_values(df):
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            series = df[column].astype("string").str.strip()
            df[column] = series.mask(series.str.lower().isin(MISSING_MARKERS))

    return df


def numeric_from_text(series):
    """Extract the first useful number from text such as ₹54,999 or 5000 mAh."""
    extracted = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def extract_brand(series):
    """The supplied dataset stores the brand at the beginning of model names."""
    result = series.astype("string").str.strip().str.split().str[0]
    result = result.replace({"<NA>": pd.NA, "nan": pd.NA})
    # Combine obvious capitalization variants.
    result = result.str.lower().replace({
        "oppo": "OPPO",
        "poco": "POCO",
        "iqoo": "iQOO",
        "itel": "itel",
        "oneplus": "OnePlus",
        "samsung": "Samsung",
        "xiaomi": "Xiaomi",
        "vivo": "Vivo",
        "realme": "Realme",
        "motorola": "Motorola",
        "apple": "Apple",
        "nokia": "Nokia",
        "google": "Google",
        "huawei": "Huawei",
        "honor": "Honor",
        "tecno": "Tecno",
        "infinix": "Infinix",
        "nothing": "Nothing",
        "sony": "Sony",
        "asus": "Asus",
        "nubia": "Nubia",
        "lava": "Lava",
        "lg": "LG",
        "gionee": "Gionee",
        "oukitel": "Oukitel",
        "ikall": "iKall",
        "jio": "Jio",
    })
    return result.str.strip()


def extract_ram_gb(series):
    return pd.to_numeric(
        series.astype("string").str.extract(r"(\d+(?:\.\d+)?)\s*(?:GB|GB RAM)", expand=False),
        errors="coerce",
    )


def extract_storage_gb(series):
    text = series.astype("string")
    extracted = text.str.extract(
        r"(\d+(?:\.\d+)?)\s*GB\s*inbuilt", expand=False
    )
    return pd.to_numeric(extracted, errors="coerce")


def extract_battery_mah(series):
    return pd.to_numeric(
        series.astype("string").str.extract(r"(\d+(?:\.\d+)?)\s*mAh", expand=False),
        errors="coerce",
    )


def extract_display_inches(series):
    return pd.to_numeric(
        series.astype("string").str.extract(
            r"(\d+(?:\.\d+)?)\s*inches?", expand=False
        ),
        errors="coerce",
    )


def extract_refresh_rate(series):
    return pd.to_numeric(
        series.astype("string").str.extract(
            r"(\d+(?:\.\d+)?)\s*Hz", expand=False
        ),
        errors="coerce",
    )


def extract_rear_camera_mp(series):
    """
    Extract the largest MP value before the front-camera section.
    Example:
    '50 MP + 48 MP + 32 MP Triple Rear & 16 MP Front Camera' -> 50.
    """
    values = []

    for value in series.astype("string"):
        if pd.isna(value):
            values.append(np.nan)
            continue

        text = str(value)
        rear_part = re.split(r"&|front", text, flags=re.IGNORECASE)[0]
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*MP", rear_part, flags=re.IGNORECASE)

        if numbers:
            values.append(max(float(n) for n in numbers))
        else:
            values.append(np.nan)

    return pd.Series(values, index=series.index, dtype="float64")


def extract_os_family(series):
    """
    The source 'os' column contains some non-OS feature descriptions.
    Keep only recognisable OS families and mark the rest as Unknown.
    """
    text = series.astype("string").str.strip()
    result = pd.Series("Unknown", index=series.index, dtype="string")

    patterns = [
        (r"^android\b", "Android"),
        (r"^ios\b", "iOS"),
        (r"^harmony\b", "HarmonyOS"),
        (r"^windows\b", "Windows"),
        (r"^kaios\b", "KaiOS"),
        (r"^symbian\b", "Symbian"),
    ]

    for pattern, label in patterns:
        result = result.mask(text.str.contains(pattern, case=False, na=False), label)

    result = result.mask(text.isna(), "Unknown")
    return result


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()


def detect_outliers(df):
    result = {}

    for column in numeric_columns(df):
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            result[column] = {
                "count": 0,
                "lower": float(q1),
                "upper": float(q3),
            }
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())

        result[column] = {
            "count": count,
            "lower": float(lower),
            "upper": float(upper),
        }

    return result


def clean_dataset(raw_df):
    """
    Cleaning is based on the actual supplied smartphone dataset:
    model, price, rating, sim, processor, ram, battery, display,
    camera, card and os.
    """
    df = standardize_column_names(raw_df.copy())
    report = {
        "original_rows": int(len(df)),
        "original_columns": int(len(df.columns)),
    }

    df = standardize_missing_values(df)

    # Dataset-specific numeric conversion.
    if "price" in df.columns:
        df["price"] = numeric_from_text(df["price"])

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Feature extraction from the text fields in this dataset.
    if "model" in df.columns:
        df["brand"] = extract_brand(df["model"])

    if "ram" in df.columns:
        df["ram_gb"] = extract_ram_gb(df["ram"])
        df["storage_gb"] = extract_storage_gb(df["ram"])

    if "battery" in df.columns:
        df["battery_mah"] = extract_battery_mah(df["battery"])

    if "display" in df.columns:
        df["display_inches"] = extract_display_inches(df["display"])
        df["refresh_rate_hz"] = extract_refresh_rate(df["display"])

    if "camera" in df.columns:
        df["rear_camera_mp"] = extract_rear_camera_mp(df["camera"])

    if "os" in df.columns:
        df["os_family"] = extract_os_family(df["os"])

    report["derived_columns"] = [
        column
        for column in [
            "brand",
            "ram_gb",
            "storage_gb",
            "battery_mah",
            "display_inches",
            "refresh_rate_hz",
            "rear_camera_mp",
            "os_family",
        ]
        if column in df.columns
    ]

    # Remove completely empty columns.
    empty_columns = [
        column for column in df.columns
        if df[column].isna().all()
    ]
    if empty_columns:
        df = df.drop(columns=empty_columns)
    report["empty_columns_removed"] = empty_columns

    # Remove exact duplicate records.
    report["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    report["missing_before"] = int(df.isna().sum().sum())

    # Impute numeric columns using median.
    numeric_imputations = {}
    for column in numeric_columns(df):
        missing = int(df[column].isna().sum())
        if missing:
            median_value = df[column].median()
            if pd.notna(median_value):
                df[column] = df[column].fillna(median_value)
                numeric_imputations[column] = {
                    "count": missing,
                    "value": float(median_value),
                }

    # Impute categorical columns using mode.
    categorical_imputations = {}
    for column in categorical_columns(df):
        missing = int(df[column].isna().sum())
        if missing:
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)
            categorical_imputations[column] = {
                "count": missing,
                "value": str(fill_value),
            }

    report["numeric_imputations"] = numeric_imputations
    report["categorical_imputations"] = categorical_imputations
    report["missing_after"] = int(df.isna().sum().sum())

    # Detect, but do not delete, outliers.
    report["outliers"] = detect_outliers(df)

    report["final_rows"] = int(len(df))
    report["final_columns"] = int(len(df.columns))

    return df, report


def get_correlation(df):
    useful = [
        column for column in [
            "price",
            "rating",
            "ram_gb",
            "storage_gb",
            "battery_mah",
            "display_inches",
            "refresh_rate_hz",
            "rear_camera_mp",
        ]
        if column in df.columns and df[column].nunique(dropna=True) > 1
    ]

    if len(useful) < 2:
        return pd.DataFrame()

    return df[useful].corr(method="pearson")


def display_name(column):
    return str(column).replace("_", " ").title()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("📱 Smartphone Intelligence")
st.sidebar.caption("DPDM Experiential Learning")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📥 Data Profiling",
        "🧹 Data Cleaning",
        "📊 Descriptive Analytics",
        "📈 Visualizations",
        "🔗 Correlation Matrix",
    ],
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Smartphone CSV",
    type=["csv"],
)


# ============================================================
# 6. LOAD DATA
# ============================================================

if uploaded_file is not None:
    file_signature = (uploaded_file.name, uploaded_file.size)

    if st.session_state.file_signature != file_signature:
        try:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file)

            if raw_df.empty:
                st.error("The uploaded CSV contains no data rows.")
                st.session_state.raw_data = None
                st.session_state.clean_data = None
            else:
                clean_df, cleaning_report = clean_dataset(raw_df)

                st.session_state.raw_data = raw_df
                st.session_state.clean_data = clean_df
                st.session_state.cleaning_report = cleaning_report
                st.session_state.file_signature = file_signature

        except Exception as exc:
            st.session_state.raw_data = None
            st.session_state.clean_data = None
            st.session_state.cleaning_report = {}
            st.error(f"CSV loading error: {exc}")

raw_df = st.session_state.raw_data
clean_df = st.session_state.clean_data


# ============================================================
# 7. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📱 Smartphone Intelligence Lab</div>
            <div class="hero-subtitle">
                Data Extraction • Cleaning • Descriptive Analytics •
                Simple, Meaningful Visualization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if clean_df is None:
        st.info("👈 Upload your smartphone CSV to begin.")

        st.markdown(
            """
            ### What this application does

            **01 — Data Import**  
            Import the smartphone CSV dataset.

            **02 — Data Profiling**  
            Examine rows, columns, data types, missing values,
            unique values and duplicates.

            **03 — Data Cleaning**  
            Standardize values, convert price and rating to numeric,
            remove duplicates and treat missing values.

            **04 — Feature Extraction**  
            Extract useful numeric fields such as brand, RAM,
            storage, battery capacity, display size and refresh rate.

            **05 — Descriptive Analytics**  
            Calculate mean, median, minimum, maximum and standard deviation.

            **06 — Visualization**  
            Use simple bar charts and distributions that directly
            answer questions about the smartphone dataset.

            **07 — Correlation**  
            Examine relationships among the cleaned numerical variables.
            """
        )

    else:
        report = st.session_state.cleaning_report

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Smartphones", f"{len(clean_df):,}")
        c2.metric("Variables", len(clean_df.columns))
        c3.metric("Duplicates Removed", report["duplicates"])
        c4.metric("Missing Values Treated", report["missing_before"])

        st.markdown("### 📋 Clean Dataset Preview")
        st.dataframe(clean_df.head(15), use_container_width=True)


# ============================================================
# 8. DATA PROFILING
# ============================================================

elif page == "📥 Data Profiling":
    st.header("📥 Data Profiling")

    if clean_df is None:
        st.warning("Upload a CSV first.")
    else:
        profile = pd.DataFrame({
            "Column": clean_df.columns,
            "Data Type": [str(clean_df[c].dtype) for c in clean_df.columns],
            "Missing Values": [int(clean_df[c].isna().sum()) for c in clean_df.columns],
            "Unique Values": [int(clean_df[c].nunique(dropna=True)) for c in clean_df.columns],
        })

        st.dataframe(profile, use_container_width=True)

        st.subheader("Statistical Profile")
        st.dataframe(
            clean_df.describe(include="all").transpose(),
            use_container_width=True,
        )


# ============================================================
# 9. DATA CLEANING
# ============================================================

elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning Laboratory")

    if clean_df is None:
        st.warning("Upload a CSV first.")
    else:
        report = st.session_state.cleaning_report

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Rows", report["original_rows"])
        c2.metric("Duplicates Removed", report["duplicates"])
        c3.metric("Missing Before", report["missing_before"])
        c4.metric("Missing After", report["missing_after"])

        st.subheader("Cleaning Pipeline")
        st.write("✓ Standardized column names")
        st.write("✓ Standardized missing-value markers")
        st.write("✓ Converted price and rating to numeric values")
        st.write("✓ Removed exact duplicate records")
        st.write("✓ Extracted useful smartphone features")
        st.write("✓ Median treatment for numeric missing values")
        st.write("✓ Mode treatment for categorical missing values")
        st.write("✓ IQR-based outlier detection without automatic deletion")

        st.subheader("Extracted Smartphone Features")
        st.write(report["derived_columns"])

        if report["empty_columns_removed"]:
            st.subheader("Completely Empty Columns Removed")
            st.write(report["empty_columns_removed"])

        st.subheader("Missing-Value Imputation")

        imputation_rows = []

        for column, info in report["numeric_imputations"].items():
            imputation_rows.append({
                "Column": column,
                "Type": "Numeric",
                "Missing Count": info["count"],
                "Replacement": round(info["value"], 4),
                "Method": "Median",
            })

        for column, info in report["categorical_imputations"].items():
            imputation_rows.append({
                "Column": column,
                "Type": "Categorical",
                "Missing Count": info["count"],
                "Replacement": info["value"],
                "Method": "Mode / Unknown",
            })

        if imputation_rows:
            st.dataframe(pd.DataFrame(imputation_rows), use_container_width=True)
        else:
            st.info("No missing values required imputation.")

        st.subheader("Outlier Report")

        outlier_rows = []
        for column, values in report["outliers"].items():
            outlier_rows.append({
                "Column": column,
                "Outliers": values["count"],
                "Lower Bound": round(values["lower"], 4),
                "Upper Bound": round(values["upper"], 4),
            })

        if outlier_rows:
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
        else:
            st.info("No numeric outlier report is available.")

        st.download_button(
            "⬇️ Download Cleaned CSV",
            clean_df.to_csv(index=False),
            "cleaned_smartphone_data.csv",
            "text/csv",
        )


# ============================================================
# 10. DESCRIPTIVE ANALYTICS
# ============================================================

elif page == "📊 Descriptive Analytics":
    st.header("📊 Descriptive Analytics")

    if clean_df is None:
        st.warning("Upload a CSV first.")
    else:
        numeric = numeric_columns(clean_df)

        if numeric:
            stats = pd.DataFrame({
                "Mean": clean_df[numeric].mean(),
                "Median": clean_df[numeric].median(),
                "Minimum": clean_df[numeric].min(),
                "Maximum": clean_df[numeric].max(),
                "Std. Deviation": clean_df[numeric].std(),
            })

            st.dataframe(stats.round(2), use_container_width=True)
        else:
            st.info("No numerical columns are available.")

        categorical = categorical_columns(clean_df)

        if categorical:
            st.subheader("Categorical Analysis")

            # Brand is much more meaningful than model for frequency analysis.
            preferred = "brand" if "brand" in categorical else categorical[0]

            category = st.selectbox(
                "Select category",
                categorical,
                index=categorical.index(preferred),
                key="descriptive_category",
            )

            counts = (
                clean_df[category]
                .astype(str)
                .value_counts()
                .head(15)
                .rename_axis(category)
                .reset_index(name="Count")
            )

            st.dataframe(counts, use_container_width=True)
        else:
            st.info("No categorical columns are available.")


# ============================================================
# 11. SIMPLE MEANINGFUL VISUALIZATIONS
# ============================================================

elif page == "📈 Visualizations":
    st.header("📈 Smartphone Brand Analysis")

    if clean_df is None:
        st.warning("Upload a CSV first.")
    else:
        st.markdown(
            """
            <div class="info-box">
            This page compares the <b>same Top 10 smartphone brands</b>
            using three simple measures: number of models, average price,
            and average rating.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Build ONE consistent Top-10 brand list.
        # Top 10 is based on the number of smartphone models.
        # All three charts use exactly these same brands.
        # ----------------------------------------------------
        if "brand" not in clean_df.columns:
            st.error("Brand information could not be extracted from the model column.")
        else:
            brand_data = clean_df.dropna(subset=["brand"]).copy()

            if brand_data.empty:
                st.info("No brand information is available in the cleaned dataset.")
            else:
                top10 = (
                    brand_data["brand"]
                    .value_counts()
                    .head(10)
                    .index
                    .tolist()
                )

                top10_df = brand_data[brand_data["brand"].isin(top10)].copy()

                summary = (
                    top10_df.groupby("brand")
                    .agg(
                        Model_Count=("brand", "size"),
                        Average_Price=("price", "mean") if "price" in top10_df.columns else ("brand", "size"),
                        Average_Rating=("rating", "mean") if "rating" in top10_df.columns else ("brand", "size"),
                    )
                    .reindex(top10)
                    .reset_index()
                    .rename(columns={"brand": "Brand"})
                )

                # ------------------------------------------------
                # Summary table
                # ------------------------------------------------
                st.subheader("📋 Top 10 Brand Summary")

                display_summary = summary.copy()

                if "Average_Price" in display_summary.columns:
                    display_summary["Average Price"] = display_summary[
                        "Average_Price"
                    ].apply(
                        lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A"
                    )

                if "Average_Rating" in display_summary.columns:
                    display_summary["Average Rating"] = display_summary[
                        "Average_Rating"
                    ].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                    )

                display_summary = display_summary[
                    ["Brand", "Model_Count", "Average Price", "Average Rating"]
                ].rename(
                    columns={
                        "Model_Count": "Number of Models",
                    }
                )

                st.dataframe(
                    display_summary,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("---")

                # ------------------------------------------------
                # 1. Top 10 brands by number of models
                # ------------------------------------------------
                count_chart = (
                    summary[
                        ["Brand", "Model_Count"]
                    ]
                    .sort_values("Model_Count", ascending=True)
                )

                fig_count = px.bar(
                    count_chart,
                    x="Model_Count",
                    y="Brand",
                    orientation="h",
                    text="Model_Count",
                    color="Model_Count",
                    color_continuous_scale="Blues",
                    title="Top 10 Smartphone Brands by Number of Models",
                )

                fig_count.update_traces(
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Number of Models: %{x}<extra></extra>"
                    ),
                )

                fig_count.update_layout(
                    template="plotly_dark",
                    height=500,
                    coloraxis_showscale=False,
                    xaxis_title="Number of Smartphone Models",
                    yaxis_title="Brand",
                    margin=dict(l=100, r=80, t=80, b=60),
                )

                st.plotly_chart(
                    fig_count,
                    use_container_width=True,
                    key="top10_brand_count",
                )

                st.caption(
                    "Top 10 brands are selected by the number of smartphone "
                    "models present in the dataset."
                )

                # ------------------------------------------------
                # 2. Average price for the SAME Top 10 brands
                # ------------------------------------------------
                st.markdown("---")

                if "price" in top10_df.columns:
                    price_chart = (
                        summary[
                            ["Brand", "Average_Price"]
                        ]
                        .dropna(subset=["Average_Price"])
                        .sort_values("Average_Price", ascending=True)
                    )

                    if not price_chart.empty:
                        fig_price = px.bar(
                            price_chart,
                            x="Average_Price",
                            y="Brand",
                            orientation="h",
                            text="Average_Price",
                            color="Average_Price",
                            color_continuous_scale="Viridis",
                            title="Average Price of the Same Top 10 Brands",
                        )

                        fig_price.update_traces(
                            texttemplate="₹%{x:,.0f}",
                            textposition="outside",
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Average Price: ₹%{x:,.0f}<extra></extra>"
                            ),
                        )

                        fig_price.update_layout(
                            template="plotly_dark",
                            height=500,
                            coloraxis_showscale=False,
                            xaxis_title="Average Price (₹)",
                            yaxis_title="Brand",
                            margin=dict(l=100, r=100, t=80, b=60),
                        )

                        st.plotly_chart(
                            fig_price,
                            use_container_width=True,
                            key="top10_brand_average_price",
                        )

                        st.caption(
                            "This chart uses the exact same Top 10 brands as the "
                            "first chart and compares their average smartphone price."
                        )
                    else:
                        st.info("No usable price values are available.")
                else:
                    st.info("Price data is not available.")

                # ------------------------------------------------
                # 3. Average rating for the SAME Top 10 brands
                # ------------------------------------------------
                st.markdown("---")

                if "rating" in top10_df.columns:
                    rating_chart = (
                        summary[
                            ["Brand", "Average_Rating"]
                        ]
                        .dropna(subset=["Average_Rating"])
                        .sort_values("Average_Rating", ascending=True)
                    )

                    if not rating_chart.empty:
                        fig_rating = px.bar(
                            rating_chart,
                            x="Average_Rating",
                            y="Brand",
                            orientation="h",
                            text="Average_Rating",
                            color="Average_Rating",
                            color_continuous_scale="Plasma",
                            title="Average Rating of the Same Top 10 Brands",
                        )

                        fig_rating.update_traces(
                            texttemplate="%{x:.1f}",
                            textposition="outside",
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Average Rating: %{x:.1f}<extra></extra>"
                            ),
                        )

                        fig_rating.update_layout(
                            template="plotly_dark",
                            height=500,
                            coloraxis_showscale=False,
                            xaxis_title="Average Rating",
                            yaxis_title="Brand",
                            xaxis=dict(range=[0, 100]),
                            margin=dict(l=100, r=80, t=80, b=60),
                        )

                        st.plotly_chart(
                            fig_rating,
                            use_container_width=True,
                            key="top10_brand_average_rating",
                        )

                        st.caption(
                            "This chart uses observed rating values for the same "
                            "Top 10 brands. Missing ratings are not treated as real ratings."
                        )
                    else:
                        st.info("No usable rating values are available.")
                else:
                    st.info("Rating data is not available.")

                


# 12. CORRELATION MATRIX
# ============================================================

elif page == "🔗 Correlation Matrix":
    st.header("🔗 Correlation Matrix")

    if clean_df is None:
        st.warning("Upload a CSV first.")
    else:
        corr = get_correlation(clean_df)

        if corr.empty:
            st.error(
                "The cleaned dataset does not contain at least two varying "
                "numerical smartphone variables."
            )
        else:
            st.success(
                f"Correlation calculated using {len(corr.columns)} numerical variables."
            )

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Pearson Correlation Matrix",
            )

            fig.update_xaxes(tickangle=-45)

            fig.update_layout(
                template="plotly_dark",
                height=max(600, len(corr.columns) * 65),
                margin=dict(l=80, r=80, t=90, b=130),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Correlation Values")
            st.dataframe(corr.round(3), use_container_width=True)

            pairs = []
            columns = corr.columns.tolist()

            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    value = corr.iloc[i, j]
                    if pd.notna(value):
                        pairs.append({
                            "Variable 1": display_name(columns[i]),
                            "Variable 2": display_name(columns[j]),
                            "Correlation": round(float(value), 3),
                            "Absolute Correlation": round(abs(float(value)), 3),
                        })

            pair_df = pd.DataFrame(pairs)

            if not pair_df.empty:
                pair_df = pair_df.sort_values(
                    "Absolute Correlation",
                    ascending=False,
                )

                st.subheader("🔎 Strongest Numerical Relationships")
                st.dataframe(
                    pair_df.head(10),
                    use_container_width=True,
                )

                st.caption(
                    "Correlation measures linear association. It does not by itself "
                    "establish cause and effect."
                )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "DPDM Experiential Learning Project | "
    "Smartphone Data Extraction, Cleaning & Descriptive Analytics"
)
