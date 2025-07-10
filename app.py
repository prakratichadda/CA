import streamlit as st
import pandas as pd
from financial_forecasting import finance_forecasting
from fraud_detection import fraud_detection_analysis
from tax_compliance import calculate_tax_liability
from invoice_processing import process_invoices
import base64
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="Your AI CHARTERED ACCOUNTANT")

st.title("Your AI CHARTERED ACCOUNTANT")
st.markdown("Choose a feature from the sidebar to begin.")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ("Financial Forecasting", "Fraud Detection", "Tax Compliance", "Invoice Processing")
)

# --- Financial Forecasting Section ---
if analysis_mode == "Financial Forecasting":
    st.header("📈 Financial Forecasting")
    st.markdown("Upload your historical sales data (CSV) to get forecasts and detect anomalies.")

    st.sidebar.subheader("Upload Data (Forecasting)")
    uploaded_file_forecast = st.sidebar.file_uploader("Upload CSV for Forecasting", type=["csv"], key="forecast_uploader")

    # --- CSV Requirements for Financial Forecasting ---
    st.markdown("""
        **CSV Requirements for Financial Forecasting:**
        - **Mandatory:** At least one numerical column representing the time-series data to be forecasted (e.g., 'sales', 'revenue', 'demand'). You will specify its name as the 'Target Column'.
        - **Highly Recommended:** A 'Date' column in a recognizable format (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY'). This helps in plotting and time-series analysis. If missing, a simple time step index will be used.
        - **Optional:** Other numerical columns (e.g., 'gdp_growth', 'unemployment_rate', 'inflation_rate', 'marketing_spend') for correlation analysis and additional visualizations.
        - **Structure:** Each row should represent a sequential time period (e.g., daily, weekly, monthly).
    """)
    # --- End CSV Requirements ---

    if uploaded_file_forecast is not None:
        file_details = {"FileName": uploaded_file_forecast.name, "FileType": uploaded_file_forecast.type}
        st.sidebar.write("File uploaded:", file_details["FileName"])

        st.sidebar.subheader("Forecasting Parameters")
        target_column_option_forecast = st.sidebar.text_input("Target Column Name", value="target_sales", help="The column in your CSV containing sales data to forecast.", key="forecast_target_col")
        date_column_option_forecast = st.sidebar.text_input("Date Column Name (Optional)", value="Date", help="If your data has a date column (e.g., 'Date', 'Month'), enter its name. If left blank or invalid, a time step index will be used.", key="forecast_date_col")
        
        forecast_months_option = st.sidebar.slider("Number of Periods to Forecast", min_value=1, max_value=24, value=12, key="forecast_months")
        contamination_option_forecast = st.sidebar.slider("Anomaly Sensitivity (Contamination)", min_value=0.001, max_value=0.1, value=0.01, format="%.3f", help="Higher value means more anomalies detected.", key="forecast_contamination")

        if st.sidebar.button("Run Financial Forecast", key="run_forecast_button"):
            with st.spinner("Processing data and running forecast... This might take a moment."):
                try:
                    df_anomalies, forecast_df, plotly_forecast_fig, plot_images_forecast = finance_forecasting(
                        uploaded_file_forecast,
                        contamination=contamination_option_forecast,
                        forecast_months=forecast_months_option,
                        target_col=target_column_option_forecast,
                        date_col=date_column_option_forecast,
                        st_object=st
                    )

                    st.success("Financial Forecasting Complete!")

                    st.subheader("📈 Sales Forecast with Anomalies")
                    st.plotly_chart(plotly_forecast_fig, use_container_width=True)

                    st.subheader("📅 Forecasted Values")
                    forecast_df_display = forecast_df.copy()
                    forecast_df_display.index.name = "Forecast Period"
                    st.dataframe(forecast_df_display)

                    st.subheader("🚨 Detected Anomalies in Historical Data")
                    if not df_anomalies[df_anomalies['is_anomaly']].empty:
                        st.dataframe(df_anomalies[df_anomalies['is_anomaly']])
                    else:
                        st.info("No anomalies detected in historical data based on the current sensitivity.")

                    # --- Display Additional Financial Forecasting Plots ---
                    st.subheader("📊 Additional Financial Visualizations")

                    # Numeric Trends (Matplotlib - downloaded as PNG)
                    if 'numeric_trends' in plot_images_forecast:
                        st.image(base64.b64decode(plot_images_forecast['numeric_trends']), caption="Numeric Trends After Cleaning")
                        st.download_button(
                            label="Download Numeric Trends Plot (PNG)",
                            data=base64.b64decode(plot_images_forecast['numeric_trends']),
                            file_name="financial_numeric_trends_plot.png",
                            mime="image/png",
                            key="download_forecast_numeric_trends_plot"
                        )
                    
                    # Sales vs Target Sales (Plotly - downloaded as HTML)
                    if 'sales_vs_target_sales_plotly' in plot_images_forecast:
                        st.plotly_chart(plot_images_forecast['sales_vs_target_sales_plotly'], use_container_width=True)
                        st.download_button(
                            label="Download Sales vs Target Sales Plot (HTML)",
                            data=plot_images_forecast['sales_vs_target_sales_plotly'].to_html().encode('utf-8'),
                            file_name="sales_vs_target_sales_plot.html",
                            mime="text/html",
                            key="download_sales_target_plot_html"
                        )

                    # Market Indicators (dropdown for selection - downloaded as HTML)
                    if 'market_indicators_plotly_figs' in plot_images_forecast and plot_images_forecast['market_indicators_plotly_figs']:
                        st.markdown("##### Individual Market Indicator Trends")
                        indicator_options = list(plot_images_forecast['market_indicators_plotly_figs'].keys())
                        if indicator_options:
                            selected_indicator = st.selectbox(
                                "Select a market indicator to view its trend:",
                                indicator_options,
                                key="forecast_market_indicator_selector"
                            )
                            selected_fig_html = plot_images_forecast['market_indicators_plotly_figs'][selected_indicator]
                            st.plotly_chart(selected_fig_html, use_container_width=True)
                            st.download_button(
                                label=f"Download {selected_indicator.replace('_', ' ').title()} Plot (HTML)",
                                data=selected_fig_html.to_html().encode('utf-8'),
                                file_name=f"{selected_indicator}_plot.html",
                                mime="text/html",
                                key=f"download_forecast_{selected_indicator}_plot_html"
                            )
                        else:
                            st.info("No market indicators found (e.g., 'gdp_growth', 'unemployment_rate', 'inflation_rate').")

                    # Correlation Heatmap (Plotly - downloaded as HTML)
                    if 'correlation_heatmap_plotly' in plot_images_forecast:
                        st.plotly_chart(plot_images_forecast['correlation_heatmap_plotly'], use_container_width=True)
                        st.download_button(
                            label="Download Correlation Heatmap Plot (HTML)",
                            data=plot_images_forecast['correlation_heatmap_plotly'].to_html().encode('utf-8'),
                            file_name="correlation_heatmap_plot.html",
                            mime="text/html",
                            key="download_correlation_heatmap_plot_html"
                        )
                    
                    # --- Download Button for Main Forecast (changed to HTML) ---
                    if plotly_forecast_fig: # Ensure plotly_forecast_fig is not None
                        st.download_button(
                            label="Download Main Forecast Plot (HTML)",
                            data=plotly_forecast_fig.to_html().encode('utf-8'),
                            file_name="financial_forecast_plot.html",
                            mime="text/html",
                            key="download_main_forecast_plot_html"
                        )

                    csv_forecast = forecast_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="Download Forecasted Data (CSV)",
                        data=csv_forecast,
                        file_name="forecasted_data.csv",
                        mime="text/csv",
                        key="download_forecast_csv"
                    )

                except Exception as e:
                    st.error(f"An error occurred during financial forecasting: {e}")
                    st.info(f"Please ensure your CSV file has a column named '{target_column_option_forecast}' for target sales and suitable numeric columns for analysis. If you specified a date column, ensure it's valid.")
    else:
        st.info("Please upload a CSV file to begin financial forecasting.")


# --- Fraud Detection Section ---
elif analysis_mode == "Fraud Detection":
    st.header("🕵️‍♂️ Fraud Detection")
    st.markdown("Upload your transaction or operational data (CSV) to identify suspicious anomalies.")

    st.sidebar.subheader("Upload Data (Fraud Detection)")
    uploaded_file_fraud = st.sidebar.file_uploader("Upload CSV for Fraud Detection", type=["csv"], key="fraud_uploader")

    # --- CSV Requirements for Fraud Detection ---
    st.markdown("""
        **CSV Requirements for Fraud Detection:**
        - **Mandatory:** At least one numerical column (e.g., `TransactionAmount`, `LoginAttempts`, `AccountBalance`) for the anomaly detection model to analyze.
        - **Highly Recommended:** A primary `Date` column (e.g., `TransactionDate`, `ActivityDate`) for time-based feature engineering and summaries. You can specify its name.
        - **Optional but Recommended:** Other relevant numerical and categorical columns that might contain patterns of normal/fraudulent behavior, such as:
            - **Numerical:** `CustomerAge`, `NumberOfItems`, `IPAddress` (if converted to numerical features), `TimeSinceLastTransaction`.
            - **Categorical (will be one-hot encoded or processed):** `TransactionType`, `Location`, `Channel`, `MerchantID`, `CustomerID`, `ProductCategory`.
        - **Structure:** Each row should represent a single event or transaction.
    """)
    # --- End CSV Requirements ---

    if uploaded_file_fraud is not None:
        file_details = {"FileName": uploaded_file_fraud.name, "FileType": uploaded_file_fraud.type}
        st.sidebar.write("File uploaded:", file_details["FileName"])

        st.sidebar.subheader("Fraud Detection Parameters")
        contamination_option_fraud = st.sidebar.slider("Anomaly Detection Sensitivity (Contamination)", min_value=0.001, max_value=0.1, value=0.01, format="%.3f", help="Higher value means more anomalies detected.", key="fraud_contamination")
        date_column_name_fraud = st.sidebar.text_input("Primary Date Column (for time-based features)", value="TransactionDate", help="If your data has a primary date column (e.g., 'TransactionDate', 'Date'), enter its name for time-based features and summaries. Leave as default if 'TransactionDate' is your primary date column.", key="fraud_date_col")

        if st.sidebar.button("Run Fraud Detection", key="run_fraud_button"):
            with st.spinner("Processing data and detecting fraud..."):
                try:
                    df_full, anomalies_df, anomaly_summary_list, top_anom_df, amount_col_name, plot_images = fraud_detection_analysis(
                        uploaded_file_fraud,
                        contamination=contamination_option_fraud,
                        date_col_name=date_column_name_fraud,
                        st_object=st
                    )

                    st.success(f"Fraud Detection Complete! Detected {len(anomalies_df)} anomalies.")

                    if not anomalies_df.empty:
                        # --- Display Plots ---
                        st.subheader("📊 Fraud Detection Visualizations")
                        
                        # Anomaly Count Plot
                        if 'anomaly_count' in plot_images:
                            st.image(base64.b64decode(plot_images['anomaly_count']), caption="Fraud vs Non-Fraud Predictions")
                            st.download_button(
                                label="Download Anomaly Count Plot (PNG)",
                                data=base64.b64decode(plot_images['anomaly_count']),
                                file_name="fraud_anomaly_count_plot.png",
                                mime="image/png",
                                key="download_anomaly_count_plot"
                            )
                        
                        # Fraud by Transaction Type
                        if 'fraud_by_type' in plot_images:
                            st.image(base64.b64decode(plot_images['fraud_by_type']), caption="Fraud by Transaction Type")
                            st.download_button(
                                label="Download Fraud by Type Plot (PNG)",
                                data=base64.b64decode(plot_images['fraud_by_type']),
                                file_name="fraud_by_type_plot.png",
                                mime="image/png",
                                key="download_fraud_by_type_plot"
                            )

                        # Fraud Over Time
                        if 'fraud_over_time' in plot_images:
                            st.image(base64.b64decode(plot_images['fraud_over_time']), caption="Fraud Predictions Over Time")
                            st.download_button(
                                label="Download Fraud Over Time Plot (PNG)",
                                data=base64.b64decode(plot_images['fraud_over_time']),
                                file_name="fraud_over_time_plot.png",
                                mime="image/png",
                                key="download_fraud_over_time_plot"
                            )

                        # Top Fraudulent Accounts
                        if 'top_fraud_accounts' in plot_images:
                            st.image(base64.b64decode(plot_images['top_fraud_accounts']), caption="Top 10 Fraudulent Accounts")
                            st.download_button(
                                label="Download Top Fraudulent Accounts Plot (PNG)",
                                data=base64.b64decode(plot_images['top_fraud_accounts']),
                                file_name="top_fraud_accounts_plot.png",
                                mime="image/png",
                                key="download_top_fraud_accounts_plot"
                            )

                        # Correlation Heatmap
                        if 'correlation_heatmap' in plot_images:
                            st.image(base64.b64decode(plot_images['correlation_heatmap']), caption="Correlation of Features with Fraud Prediction")
                            st.download_button(
                                label="Download Correlation Heatmap Plot (PNG)",
                                data=base64.b64decode(plot_images['correlation_heatmap']),
                                file_name="fraud_correlation_heatmap.png",
                                mime="image/png",
                                key="download_correlation_heatmap"
                            )

                        st.subheader("🚨 Detected Anomalies Data")
                        st.dataframe(anomalies_df)

                        st.subheader("📝 Anomaly Summary")
                        for summary_item in anomaly_summary_list:
                            st.write(summary_item)
                        
                        if top_anom_df is not None and not top_anom_df.empty:
                            st.subheader(f"💸 Top 5 Anomalies by {amount_col_name}")
                            st.dataframe(top_anom_df)
                        elif top_anom_df is not None and top_anom_df.empty:
                             st.info("No anomalies detected to show top anomalies.")
                        else: # top_anom_df is None
                            st.info("Could not identify a suitable numerical column to sort top anomalies or no anomalies detected.")

                        csv_anomalies = anomalies_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Detected Anomalies Data (CSV)",
                            data=csv_anomalies,
                            file_name="detected_anomalies.csv",
                            mime="text/csv",
                            key="download_anomalies_csv"
                        )
                    else:
                        st.info("No anomalies detected based on the current sensitivity and data.")


                except Exception as e:
                    st.error(f"An error occurred during fraud detection: {e}")
                    st.info("Please ensure your CSV contains numerical data for detection. If you provided a date column name, ensure it's correct and in a recognizable format.")
    else:
        st.info("Please upload a CSV file to begin fraud detection.")


# --- Tax Compliance Section ---
elif analysis_mode == "Tax Compliance":
    st.header("💰 Tax Compliance Calculator (India)")
    st.markdown("Calculate your income tax liability based on gross income, deductions, and the tax year.")
    st.markdown("*(Note: This calculator uses simplified Indian Old Tax Regime slabs for individuals. Actual tax calculations can be more complex with various exemptions, surcharges, and different regimes.)*")

    st.sidebar.subheader("Tax Input Parameters")
    income_input = st.sidebar.number_input("Gross Annual Income (₹)", min_value=0.0, value=750000.0, step=10000.0, format="%.2f", key="tax_income_input")
    deductions_input = st.sidebar.number_input("Eligible Deductions (₹)", min_value=0.0, value=150000.0, step=5000.0, format="%.2f", help="e.g., 80C, 80D, Standard Deduction, etc. (relevant for old regime)", key="tax_deductions_input")
    tax_year_input = st.sidebar.selectbox("Tax Year", options=[2024, 2023], index=0, help="2024 refers to FY 2023-24 / AY 2024-25", key="tax_year_input")

    # No CSV upload for this feature, so no specific CSV requirements here.
    # st.info("This feature does not require a CSV upload. Please use the input fields in the sidebar.")

    if st.sidebar.button("Calculate Tax Liability", key="calculate_tax_button"):
        with st.spinner("Calculating tax liability..."):
            try:
                tax_result = calculate_tax_liability(
                    income=income_input,
                    deductions=deductions_input,
                    year=tax_year_input
                )

                st.success("Tax Liability Calculated!")

                st.subheader("Summary of Tax Calculation")
                st.write(f"**Gross Annual Income:** ₹{tax_result['gross_income']:,.2f}")
                st.write(f"**Total Eligible Deductions:** ₹{tax_result['total_deductions']:,.2f}")
                st.write(f"**Taxable Income:** ₹{tax_result['taxable_income']:,.2f}")
                st.write(f"**Tax Before Cess:** ₹{tax_result['tax_before_cess']:,.2f}")
                st.write(f"**Health & Education Cess (4%):** ₹{tax_result['cess']:,.2f}")
                st.write(f"**Total Tax Liability:** ₹{tax_result['total_tax_liability']:,.2f}")

                st.subheader("Tax Slab Breakdown")
                if tax_result['tax_breakdown']:
                    breakdown_df = pd.DataFrame(tax_result['tax_breakdown'])
                    st.dataframe(breakdown_df.style.format({
                        'amount_in_slab': "₹{:,.2f}",
                        'tax_in_segment': "₹{:,.2f}"
                    }))

                    # Plotting the tax breakdown as a bar chart
                    fig = px.bar(
                        breakdown_df,
                        x='slab_range',
                        y='tax_in_segment',
                        color='rate',
                        title='Tax Contribution per Slab',
                        labels={'slab_range': 'Tax Slab', 'tax_in_segment': 'Tax Amount (₹)', 'rate': 'Tax Rate'},
                        height=400
                    )
                    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray': breakdown_df['slab_range'].tolist()})
                    st.plotly_chart(fig, use_container_width=True)

                    # Download button for tax breakdown
                    csv_tax_breakdown = breakdown_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Tax Breakdown (CSV)",
                        data=csv_tax_breakdown,
                        file_name="tax_breakdown.csv",
                        mime="text/csv",
                        key="download_tax_breakdown_csv"
                    )

                else:
                    st.info("No tax liability calculated for the given income and deductions (likely due to income being below taxable limits).")

            except ValueError as e:
                st.error(f"Error in tax calculation: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- Invoice Processing Section ---
elif analysis_mode == "Invoice Processing":
    st.header("🧾 Invoice Processing & Analysis")
    st.markdown("Upload your invoice data (CSV) to perform customer segmentation, fraud detection, and budget vs. actual analysis.")

    st.sidebar.subheader("Upload Data (Invoice Processing)")
    uploaded_file_invoice = st.sidebar.file_uploader("Upload CSV for Invoice Processing", type=["csv"], key="invoice_uploader")

    # --- CSV Requirements for Invoice Processing ---
    st.markdown("""
        **CSV Requirements for Invoice Processing:**
        - **Mandatory columns:**
            - `invoice_date` (Date of the invoice, e.g., 'YYYY-MM-DD')
            - `amount` (Numerical value of the invoice item)
            - `product_id` (Identifier for the product/service)
        - **Recommended/Optional columns for full analysis:**
            - `invoice_id` (Unique ID for the invoice)
            - `first_name`, `last_name` (Client's name)
            - `email` (Client's email address)
            - `city` (Client's city for geographical segmentation)
            - `job` (Client's job role for occupational segmentation and budget analysis)
            - `qty` (Quantity of the product/service)
            - `product_name`, `category` (Additional product details)
        - **Structure:** Each row should ideally represent a single line item on an invoice.
    """)
    # --- End CSV Requirements ---

    if uploaded_file_invoice is not None:
        if st.sidebar.button("Process Invoices", key="process_invoices_button"):
            with st.spinner("Processing invoice data... This might take a moment."):
                try:
                    df_original, top_segments_df, city_revenue_fig, revenue_trend_fig, \
                    suspicious_invoices_df, extracted_entities_df, actual_vs_budget_df, audit_flags_df = process_invoices(uploaded_file_invoice)

                    st.success("Invoice Processing Complete!")

                    st.subheader("Summary of Processed Invoices")
                    st.write(f"Total Invoices Processed: **{len(df_original)}**")
                    st.write(f"Total Revenue: **₹{df_original['total_value'].sum():,.2f}**")
                    st.write("---")

                    # --- Customer Segmentation ---
                    st.subheader("👥 Customer Segmentation Analysis")
                    st.markdown("##### Top Customer Segments by Revenue")
                    if not top_segments_df.empty:
                        st.dataframe(top_segments_df.style.format({
                            'total_revenue': "₹{:,.2f}",
                            'avg_invoice_amount': "₹{:,.2f}"
                        }))
                        st.download_button(
                            label="Download Top Segments (CSV)",
                            data=top_segments_df.to_csv(index=False).encode('utf-8'),
                            file_name="top_customer_segments.csv",
                            mime="text/csv",
                            key="download_top_segments_csv"
                        )
                    else:
                        st.info("No customer segments found. Ensure 'city' and 'job' columns are present and data is sufficient.")

                    st.markdown("##### Revenue by City")
                    st.plotly_chart(city_revenue_fig, use_container_width=True)
                    st.download_button(
                        label="Download Revenue by City Plot (HTML)",
                        data=city_revenue_fig.to_html().encode('utf-8'),
                        file_name="revenue_by_city_plot.html",
                        mime="text/html",
                        key="download_revenue_by_city_html"
                    )

                    st.markdown("##### Monthly Revenue Trend")
                    st.plotly_chart(revenue_trend_fig, use_container_width=True)
                    st.download_button(
                        label="Download Monthly Revenue Trend Plot (HTML)",
                        data=revenue_trend_fig.to_html().encode('utf-8'),
                        file_name="monthly_revenue_trend_plot.html",
                        mime="text/html",
                        key="download_monthly_revenue_trend_html"
                    )

                    # --- Fraud Detection ---
                    st.subheader("🕵️‍♂️ Invoice Fraud Detection")
                    if not suspicious_invoices_df.empty:
                        st.warning(f"Detected **{len(suspicious_invoices_df)}** potentially suspicious invoices!")
                        st.dataframe(suspicious_invoices_df.style.format({'amount': "₹{:,.2f}"}))
                        st.download_button(
                            label="Download Suspicious Invoices (CSV)",
                            data=suspicious_invoices_df.to_csv(index=False).encode('utf-8'),
                            file_name="suspicious_invoices.csv",
                            mime="text/csv",
                            key="download_suspicious_invoices_csv"
                        )
                    else:
                        st.info("No suspicious invoices detected based on rules and ML model. Ensure 'amount', 'invoice_date', 'first_name', and 'product_id' columns are present and correctly formatted.")

                    # --- Named Entity Extraction ---
                    st.subheader("📝 Extracted Invoice Entities")
                    st.info("These are key details extracted from your invoice data for quick review.")
                    if not extracted_entities_df.empty:
                        st.dataframe(extracted_entities_df)
                        st.download_button(
                            label="Download Extracted Entities (CSV)",
                            data=extracted_entities_df.to_csv(index=False).encode('utf-8'),
                            file_name="extracted_invoice_entities.csv",
                            mime="text/csv",
                            key="download_extracted_entities_csv"
                        )
                    else:
                        st.info("No entities extracted. Please ensure columns like 'first_name', 'last_name', 'email', 'city', 'job', 'amount', 'product_id' are present.")


                    # --- Budget vs. Actual Analysis ---
                    st.subheader("📊 Budget vs. Actual Analysis (by Job Role)")
                    st.info("Compares actual expenses (from invoices) against simulated budget values for each job role.")
                    if not actual_vs_budget_df.empty:
                        st.dataframe(actual_vs_budget_df.style.format({
                            'actual': "₹{:,.2f}",
                            'budget': "₹{:,.2f}",
                            'variance': "₹{:,.2f}"
                        }))
                        st.download_button(
                            label="Download Budget vs. Actual (CSV)",
                            data=actual_vs_budget_df.to_csv(index=False).encode('utf-8'),
                            file_name="budget_vs_actual.csv",
                            mime="text/csv",
                            key="download_budget_vs_actual_csv"
                        )

                        # Plot Budget vs Actual
                        fig_budget = px.bar(
                            actual_vs_budget_df,
                            x='job',
                            y=['actual', 'budget'],
                            barmode='group',
                            title='Actual vs. Budget by Job Role',
                            labels={'value': 'Amount (₹)', 'job': 'Job Role'},
                            height=500
                        )
                        st.plotly_chart(fig_budget, use_container_width=True)
                        st.download_button(
                            label="Download Budget vs. Actual Plot (HTML)",
                            data=fig_budget.to_html().encode('utf-8'),
                            file_name="budget_vs_actual_plot.html",
                            mime="text/html",
                            key="download_budget_vs_actual_plot_html"
                        )

                    else:
                        st.info("No data available for Budget vs. Actual analysis. Ensure 'amount', 'invoice_date', and 'job' columns are present.")


                    st.markdown("##### Audit Flags for Further Review")
                    if not audit_flags_df.empty:
                        st.dataframe(audit_flags_df.style.format({'amount': "₹{:,.2f}"}))
                        st.download_button(
                            label="Download Audit Flags (CSV)",
                            data=audit_flags_df.to_csv(index=False).encode('utf-8'),
                            file_name="invoice_audit_flags.csv",
                            mime="text/csv",
                            key="download_audit_flags_csv"
                        )
                    else:
                        st.info("No specific audit flags raised (e.g., duplicates, high value invoices).")


                except ValueError as e:
                    st.error(f"Data processing error for invoices: {e}. Please check your CSV column names and data types as per the requirements below.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during invoice processing: {e}. Please check your CSV file's format.")
    else:
        st.info("Please upload a CSV file to begin invoice processing.")