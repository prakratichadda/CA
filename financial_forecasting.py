import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px # For interactive plots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Added StandardScaler for Z-score or general scaling
from scipy.stats import zscore # For outlier detection
import datetime
import io
import base64
from keras.models import Sequential
from keras.layers import LSTM, Dense
# Import streamlit for st.info messages, if you are passing it from app.py
# If not passing st_object, remove 'st_object=None' from function signatures
# and replace 'if st_object: st_object.info(...)' with 'print(...)' or remove
# from app.py: import streamlit as st

def finance_forecasting(filepath: str, contamination: float = 0.01, forecast_months: int = 12, 
                        target_col: str = 'target_sales', date_col: str = 'Date', st_object=None):
    """
    Main function to perform financial forecasting, anomaly detection, and visualization.

    Args:
        filepath (str): Path to the CSV data file.
        contamination (float): The proportion of outliers in the data set for IsolationForest.
        forecast_months (int): Number of future periods (steps/months) to forecast.
        target_col (str): The name of the column to forecast.
        date_col (str): The name of the date column in the CSV. If not found or invalid,
                        a numerical time step index will be used.
        st_object (streamlit): The streamlit object for displaying messages.
    
    Returns:
        tuple: (df_anomalies, forecast_df, plotly_forecast_fig, plot_images)
            - df_anomalies (pd.DataFrame): DataFrame with detected anomalies.
            - forecast_df (pd.DataFrame): DataFrame with forecasted values.
            - plotly_forecast_fig (go.Figure): Plotly figure object for interactive forecast visualization.
            - plot_images (dict): Dictionary of base64 encoded plot images for other visualizations.
    """
    plot_images = {} # Dictionary to store base64 encoded plot images

    # ------------------ Data Loading and Preparation ------------------ #
    def load_and_prepare_data(fp: str, dc: str) -> pd.DataFrame:
        """
        Loads CSV data, cleans it, handles missing values, duplicates, and sets up time index.
        """
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip() # Clean column names

        # Handle Date column or create a time_step index
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors='coerce')
            df = df.dropna(subset=[dc]) # Drop rows where date conversion failed
            if not df.empty:
                df = df.sort_values(dc)
                df = df.set_index(dc)
                if st_object: st_object.info(f"Using '{dc}' as date column for time series analysis.")
            else:
                if st_object: st_object.warning(f"Date column '{dc}' found but all values are invalid/missing after conversion. Falling back to numerical time steps.")
                df['time_step'] = range(len(df))
                df = df.set_index('time_step')
        else:
            if st_object: st_object.warning(f"Date column '{dc}' not found. Using numerical time steps for analysis.")
            df['time_step'] = range(len(df))
            df = df.set_index('time_step')
        
        if df.empty:
            raise ValueError("No valid data rows found after date processing.")

        # Fill missing values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True) # Fallback for any remaining NaNs (e.g., at the very start)

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Ensure the target column exists and is numeric
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the uploaded CSV.")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(df[target_col].mean() if pd.api.types.is_numeric_dtype(df[target_col]) else 0)

        return df

    # ------------------ Anomaly Detection ------------------ #
    def detect_anomalies(df: pd.DataFrame, col='target_sales', contam=0.01) -> pd.DataFrame:
        """
        Detects anomalies in the specified target column (and potentially other numeric features)
        using IsolationForest.
        """
        # Automatically detect numeric columns for anomaly detection
        numeric_cols_for_anomaly = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols_for_anomaly:
            if st_object: st_object.warning("No numeric columns found for IsolationForest anomaly detection. Skipping.")
            df['is_anomaly'] = 0 # No anomalies if no numeric data
            return df

        scaler = StandardScaler() # Use StandardScaler for IsolationForest features
        X_scaled = scaler.fit_transform(df[numeric_cols_for_anomaly])

        model = IsolationForest(contamination=contam, random_state=42)
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
        df_copy['anomaly'] = model.fit_predict(X_scaled)
        df_copy['is_anomaly'] = df_copy['anomaly'] == -1 # -1 is anomaly, 1 is normal
        return df_copy

    # ------------------ Forecasting ------------------ #
    def forecast_target(df: pd.DataFrame, col='target_sales', f_months=12) -> pd.DataFrame:
        """
        Forecasts future values of the target column using an LSTM model.
        Handles both DatetimeIndex and numerical index for future periods.
        """
        data_to_scale = df[[col]].dropna()

        if data_to_scale.empty:
            raise ValueError(f"No valid data in target column '{col}' for forecasting after dropping NaNs.")
            
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_to_scale)

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        SEQ_LEN = 12 # Sequence length for LSTM
        if len(scaled_data) < SEQ_LEN + 1:
            raise ValueError(f"Not enough data to create sequences for forecasting. Need at least {SEQ_LEN + 1} data points for LSTM (current: {len(scaled_data)}).")

        X, y = create_sequences(scaled_data, SEQ_LEN)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and compile LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(SEQ_LEN, 1)), # Changed to relu as per your script
            LSTM(50, activation='relu'), # Changed to relu
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        try:
            model.fit(X, y, epochs=30, batch_size=16, verbose=0)
        except Exception as e:
            raise RuntimeError(f"Error during LSTM model training: {e}. This might be due to insufficient or poorly formatted data.")

        # Forecast future values
        input_seq = scaled_data[-SEQ_LEN:] # Start with the last SEQ_LEN data points
        forecast = []
        for _ in range(f_months):
            input_reshaped = input_seq.reshape((1, SEQ_LEN, 1))
            pred = model.predict(input_reshaped, verbose=0)
            forecast.append(pred[0, 0])
            input_seq = np.append(input_seq[1:], pred)

        # Inverse transform the forecast to original scale
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        # Create future index based on original df's index type
        if isinstance(df.index, pd.DatetimeIndex):
            # Use MonthEnd for consistency with freq='M' in example, or MonthBegin if that's more appropriate
            last_date_in_df = df.index[-1]
            future_index = pd.date_range(start=last_date_in_df + pd.offsets.MonthBegin(1), periods=f_months, freq='M')
        else: # Numerical time step index
            last_time_step_in_df = df.index[-1]
            future_index = range(last_time_step_in_df + 1, last_time_step_in_df + 1 + f_months)
        
        forecast_df = pd.DataFrame(forecast, index=future_index, columns=[f'Forecast_{col}'])
        return forecast_df

    # ------------------ Plotting Functions ------------------ #
    def get_base64_image(plt_figure):
        """Converts a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        plt_figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(plt_figure) # Close the figure to free memory
        return img_base64

    def plot_numeric_trends(df_input, numeric_cols_to_plot):
        if df_input.empty or not numeric_cols_to_plot:
            if st_object: st_object.warning("No data or numeric columns to plot numeric trends.")
            return None
        
        fig, ax = plt.subplots(figsize=(15, 8))
        df_input[numeric_cols_to_plot].plot(ax=ax)
        ax.set_title("Numeric Trends After Cleaning")
        ax.set_xlabel("Date" if isinstance(df_input.index, pd.DatetimeIndex) else "Time Step")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        return fig

    def plot_sales_vs_target_sales(df_input, sales_col, target_sales_col):
        if sales_col not in df_input.columns or target_sales_col not in df_input.columns:
            if st_object: st_object.warning(f"Cannot plot Sales vs Target Sales: '{sales_col}' or '{target_sales_col}' column not found.")
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_input.index, y=df_input[sales_col], mode='lines+markers', name='Sales',
            line=dict(color='teal', width=2), marker=dict(symbol="circle", size=4)
        ))
        fig.add_trace(go.Scatter(
            x=df_input.index, y=df_input[target_sales_col], mode='lines+markers', name='Target Sales',
            line=dict(color='orange', width=2, dash='dot'), marker=dict(symbol="star", size=4)
        ))
        fig.update_layout(
            title="Sales vs Target Sales (Interactive)",
            xaxis_title="Date" if isinstance(df_input.index, pd.DatetimeIndex) else "Time Step",
            yaxis_title="Amount",
            hovermode='x unified',
            plot_bgcolor='white',
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig

    def plot_market_indicators_treemap(df_input):
        market_indicators = [col for col in ['gdp_growth', 'unemployment_rate', 'inflation_rate'] if col in df_input.columns]
        
        if not market_indicators:
            if st_object: st_object.warning("No market indicators (gdp_growth, unemployment_rate, inflation_rate) found to plot.")
            return None

        # Create a DataFrame for the treemap, summing values (or latest values)
        # For a treemap, we usually need hierarchical data. For simple indicators,
        # we can just show their current or average values.
        # A bar chart or line chart for individual indicators is more appropriate.
        # Let's use a simpler approach: create a bar chart for individual indicator trends
        # which is what your example hinted at with the dropdown.
        # Instead of Treemap, let's make a single plot for one indicator or use subplots.

        # Returning a dictionary of plotly figures for individual indicators to be displayed
        # with st.plotly_chart in app.py with a dropdown.
        indicator_figs = {}
        for indicator in market_indicators:
            fig = px.line(df_input, x=df_input.index, y=indicator, title=f"{indicator.replace('_', ' ').title()} Over Time")
            fig.update_layout(
                xaxis_title="Date" if isinstance(df_input.index, pd.DatetimeIndex) else "Time Step",
                yaxis_title="Value",
                hovermode='x unified',
                plot_bgcolor='whitesmoke',
                margin=dict(l=40, r=40, t=80, b=40)
            )
            indicator_figs[indicator] = fig
        return indicator_figs
        
    def plot_correlation_heatmap(df_input, features_used_for_corr):
        # Ensure only numeric columns are selected for correlation
        numeric_df = df_input[features_used_for_corr].select_dtypes(include=np.number)

        if numeric_df.empty:
            if st_object: st_object.warning("No numerical features available for correlation heatmap.")
            return None

        corr = numeric_df.corr()
        
        # Create an interactive Plotly heatmap instead of static Matplotlib
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto", # Allows cells to stretch to fit text
            color_continuous_scale="Viridis",
            title="Correlation Heatmap of Financial Indicators"
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        return fig


    # Main execution logic
    try:
        if st_object: st_object.info("Loading and preparing financial data...")
        df_cleaned = load_and_prepare_data(filepath, date_col)
        # Identify numeric columns for general plotting (excluding potentially 'anomaly' flags)
        numeric_cols_for_general_plots = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        # Ensure sales and target_sales are in the list if they are numeric
        if 'sales' not in numeric_cols_for_general_plots and 'sales' in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned['sales']):
            numeric_cols_for_general_plots.append('sales')
        if target_col not in numeric_cols_for_general_plots and target_col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[target_col]):
            numeric_cols_for_general_plots.append(target_col)
        
        # Remove anomaly-related columns if they are picked up, for the general trend plot
        numeric_cols_for_general_plots = [col for col in numeric_cols_for_general_plots if col not in ['anomaly', 'is_anomaly']]

    except ValueError as e:
        raise ValueError(f"Data loading and preparation failed: {e}")

    df_anomalies = None # Initialize to None
    try:
        if st_object: st_object.info("Detecting anomalies in financial data...")
        df_anomalies = detect_anomalies(df_cleaned, col=target_col, contam=contamination)
    except Exception as e:
        if st_object: st_object.error(f"Anomaly detection for financial forecasting failed: {e}. Proceeding with forecasting without anomaly flags.")
        # If anomaly detection fails, continue without 'is_anomaly' column
        df_anomalies = df_cleaned.copy()
        df_anomalies['is_anomaly'] = False # Default to no anomalies


    forecast_df = pd.DataFrame() # Initialize to empty
    plotly_forecast_fig = go.Figure() # Initialize to empty figure

    try:
        if st_object: st_object.info(f"Forecasting {target_col} for {forecast_months} future periods...")
        forecast_df = forecast_target(df_anomalies, col=target_col, f_months=forecast_months)

        # Generate the main forecast plot (Plotly)
        plotly_forecast_fig = go.Figure()
        
        # Plot historical data
        plotly_forecast_fig.add_trace(go.Scatter(x=df_anomalies.index, y=df_anomalies[target_col], 
                                                name='Historical Sales', mode='lines+markers', line=dict(color='blue')))
        
        # Plot anomalies if detected
        if 'is_anomaly' in df_anomalies.columns and df_anomalies['is_anomaly'].any():
            anomalies_to_plot = df_anomalies[df_anomalies['is_anomaly']]
            plotly_forecast_fig.add_trace(go.Scatter(x=anomalies_to_plot.index, y=anomalies_to_plot[target_col], 
                                                     mode='markers', name='Anomalies', 
                                                     marker=dict(color='red', size=8, symbol='x')))

        # Plot forecasted data
        plotly_forecast_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[forecast_df.columns[0]], 
                                                name='Forecasted Sales', mode='lines+markers', 
                                                line=dict(color='orange', dash='dash')))

        plotly_forecast_fig.update_layout(
            title=f'Historical and Forecasted {target_col} with Anomalies',
            xaxis_title="Date" if isinstance(df_anomalies.index, pd.DatetimeIndex) else "Time Step",
            yaxis_title=target_col,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
            margin=dict(l=40, r=40, t=80, b=40)
        )

    except (ValueError, RuntimeError) as e:
        if st_object: st_object.error(f"Forecasting failed: {e}. Please check your data and parameters. Displaying historical data only.")
        # If forecasting fails, still prepare the historical plot for display
        plotly_forecast_fig = go.Figure()
        plotly_forecast_fig.add_trace(go.Scatter(x=df_anomalies.index, y=df_anomalies[target_col], 
                                                name='Historical Sales', mode='lines+markers', line=dict(color='blue')))
        if 'is_anomaly' in df_anomalies.columns and df_anomalies['is_anomaly'].any():
            anomalies_to_plot = df_anomalies[df_anomalies['is_anomaly']]
            plotly_forecast_fig.add_trace(go.Scatter(x=anomalies_to_plot.index, y=anomalies_to_plot[target_col], 
                                                     mode='markers', name='Anomalies', 
                                                     marker=dict(color='red', size=8, symbol='x')))
        plotly_forecast_fig.update_layout(
            title=f'Historical {target_col} (Forecasting Failed)',
            xaxis_title="Date" if isinstance(df_anomalies.index, pd.DatetimeIndex) else "Time Step",
            yaxis_title=target_col,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
            margin=dict(l=40, r=40, t=80, b=40)
        )


    # --- Generate Additional Plots ---
    if st_object: st_object.info("Generating additional visualizations for financial forecasting...")
    
    # Plot 1: Numeric Trends
    fig_numeric_trends = plot_numeric_trends(df_cleaned, numeric_cols_for_general_plots)
    if fig_numeric_trends: plot_images['numeric_trends'] = get_base64_image(fig_numeric_trends)

    # Plot 2: Sales vs Target Sales (Plotly)
    if 'sales' in df_cleaned.columns and target_col in df_cleaned.columns:
        plotly_sales_chart = plot_sales_vs_target_sales(df_cleaned, 'sales', target_col)
        plot_images['sales_vs_target_sales_plotly'] = plotly_sales_chart # Keep as Plotly object
    
    # Plot 3: Market Indicators (returns a dict of plotly figs)
    plotly_market_indicator_figs = plot_market_indicators_treemap(df_cleaned) # Renamed from treemap to be generic
    if plotly_market_indicator_figs: plot_images['market_indicators_plotly_figs'] = plotly_market_indicator_figs # Store dict of plotly figs

    # Plot 4: Correlation Heatmap (Plotly)
    # Ensure all numeric columns are considered for correlation
    all_numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    if 'anomaly' in all_numeric_cols: all_numeric_cols.remove('anomaly') # Remove internal anomaly flag
    plotly_correlation_heatmap_fig = plot_correlation_heatmap(df_cleaned, all_numeric_cols)
    if plotly_correlation_heatmap_fig: plot_images['correlation_heatmap_plotly'] = plotly_correlation_heatmap_fig # Keep as Plotly object


    return df_anomalies, forecast_df, plotly_forecast_fig, plot_images