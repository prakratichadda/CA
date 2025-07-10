# fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
# import streamlit as st # Only import if you want to use st directly within this file, otherwise pass it as st_object

def fraud_detection_analysis(file_path, contamination=0.01, date_col_name='TransactionDate', st_object=None):
    """
    Main function for fraud detection analysis.

    Args:
        file_path (str): Path to the CSV data file.
        contamination (float): The proportion of outliers in the data set for IsolationForest.
        date_col_name (str): The name of the primary date column for time-based features (e.g., 'TransactionDate').
        st_object (streamlit): The Streamlit object for displaying messages.

    Returns:
        tuple: (df, anomalies_df, anomaly_summary, top_anom_df, amount_col_name, plot_base64_images)
            - df (pd.DataFrame): Original DataFrame with anomaly flags.
            - anomalies_df (pd.DataFrame): DataFrame containing only detected anomalies.
            - anomaly_summary (list): List of summary strings for anomalies.
            - top_anom_df (pd.DataFrame or None): Top anomalies by value.
            - amount_col_name (str or None): The name of the identified amount column.
            - plot_base64_images (dict): Dictionary of base64 encoded plot images.
    """

    def load_data(fp):
        """Loads CSV data."""
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip() # Clean column names
        return df

    def feature_engineer_fraud_data(df, primary_date_col): # Removed st_object from here, pass through main func
        """
        Engineers new features relevant for fraud detection from raw transaction data.
        Assumes existence of columns like 'TransactionDate', 'PreviousTransactionDate',
        'TransactionAmount', 'LoginAttempts', 'AccountBalance', 'TransactionType',
        'Location', 'Channel', 'CustomerOccupation', 'CustomerAge', 'TransactionDuration'.
        """
        df_copy = df.copy() # Work on a copy

        # Convert date columns (handle gracefully if they don't exist)
        for col in [primary_date_col, 'PreviousTransactionDate']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                # Fill missing dates if they are critical for features below
                # For TimeSinceLastTransaction, fillna(-1) will be used for calculation
            # else:
                # if st_object: st_object.warning(f"Column '{col}' not found. Related features might be skipped or impacted.")

        # --- Feature Engineering ---

        # 1. Feature: Time since last transaction
        if primary_date_col in df_copy.columns and 'PreviousTransactionDate' in df_copy.columns:
            df_copy['TimeSinceLastTransaction'] = (df_copy[primary_date_col] - df_copy['PreviousTransactionDate']).dt.total_seconds()
            df_copy['TimeSinceLastTransaction'].fillna(-1, inplace=True) # Fill NaNs (e.g., first transaction)
        else:
            df_copy['TimeSinceLastTransaction'] = -1 # Default value if columns are missing
            # if st_object: st_object.info("Skipping 'TimeSinceLastTransaction' feature due to missing date columns.")

        # 2. Feature: Transaction hour and weekday
        if primary_date_col in df_copy.columns:
            df_copy['TransactionHour'] = df_copy[primary_date_col].dt.hour.fillna(-1)
            df_copy['TransactionWeekday'] = df_copy[primary_date_col].dt.weekday.fillna(-1)
        else:
            df_copy['TransactionHour'] = -1
            df_copy['TransactionWeekday'] = -1
            # if st_object: st_object.info("Skipping 'TransactionHour' and 'TransactionWeekday' features due to missing primary date column.")


        # 3. Feature: Is night transaction (12 AM to 6 AM)
        df_copy['IsNightTransaction'] = df_copy['TransactionHour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)


        # 4. Feature: High transaction amount (above 95th percentile)
        if 'TransactionAmount' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['TransactionAmount']):
            amount_threshold = df_copy['TransactionAmount'].quantile(0.95)
            df_copy['HighTransactionAmount'] = (df_copy['TransactionAmount'] > amount_threshold).astype(int)
        else:
            df_copy['HighTransactionAmount'] = 0 # Default if no TransactionAmount
            # if st_object: st_object.info("Skipping 'HighTransactionAmount' feature due to missing or non-numeric 'TransactionAmount'.")


        # 5. Feature: High login attempts
        if 'LoginAttempts' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['LoginAttempts']):
            df_copy['HighLoginAttempts'] = (df_copy['LoginAttempts'] > 3).astype(int)
        else:
            df_copy['HighLoginAttempts'] = 0 # Default
            # if st_object: st_object.info("Skipping 'HighLoginAttempts' feature due to missing or non-numeric 'LoginAttempts'.")


        # 6. Feature: Low account balance (below 100)
        if 'AccountBalance' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['AccountBalance']):
            df_copy['LowAccountBalance'] = (df_copy['AccountBalance'] < 100).astype(int)
        else:
            df_copy['LowAccountBalance'] = 0 # Default
            # if st_object: st_object.info("Skipping 'LowAccountBalance' feature due to missing or non-numeric 'AccountBalance'.")


        # 7. Feature: Transaction amount to balance ratio
        if 'TransactionAmount' in df_copy.columns and 'AccountBalance' in df_copy.columns \
           and pd.api.types.is_numeric_dtype(df_copy['TransactionAmount']) and pd.api.types.is_numeric_dtype(df_copy['AccountBalance']):
            # Add 1 to balance to avoid division by zero or very small numbers causing huge ratios
            # Replace 0s in AccountBalance to avoid division by zero before adding 1
            df_copy['TransactionAmountToBalanceRatio'] = df_copy['TransactionAmount'] / (df_copy['AccountBalance'].replace(0, np.nan) + 1).fillna(1)
        else:
            df_copy['TransactionAmountToBalanceRatio'] = 0 # Default
            # if st_object: st_object.info("Skipping 'TransactionAmountToBalanceRatio' feature due to missing or non-numeric 'TransactionAmount'/'AccountBalance'.")


        # 8. Encode categorical columns
        categorical_cols_to_encode = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
        encoded_cols_names = [] # To keep track of actually encoded columns
        for col in categorical_cols_to_encode:
            if col in df_copy.columns:
                try:
                    df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))
                    encoded_cols_names.append(col)
                except Exception as e:
                    if st_object: st_object.warning(f"Could not encode categorical column '{col}': {e}")
            # else:
                # if st_object: st_object.info(f"Categorical column '{col}' not found, skipping encoding.")
        
        # --- Define the final set of features to be used by the model ---
        base_features = [
            'TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts',
            'AccountBalance'
        ]
        
        engineered_features = [
            'TimeSinceLastTransaction', 'TransactionHour',
            'TransactionWeekday', 'IsNightTransaction', 'HighTransactionAmount',
            'HighLoginAttempts', 'LowAccountBalance', 'TransactionAmountToBalanceRatio'
        ]
        
        final_features = []
        for feature in base_features + engineered_features + encoded_cols_names:
            if feature in df_copy.columns:
                final_features.append(feature)
            # else:
            #     if st_object: st_object.warning(f"Feature '{feature}' not found in data for modeling.")

        # Fill any remaining NaNs in the selected features that might be generated by feature engineering
        # Use mean for more robust filling of numerical data for features
        for f_col in final_features:
            if df_copy[f_col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_copy[f_col]):
                    df_copy[f_col].fillna(df_copy[f_col].mean(), inplace=True)
                else:
                    df_copy[f_col].fillna(0, inplace=True) # Fallback for non-numeric or if mean is NaN

        return df_copy, final_features

    def detect_anomalies(df_input, features_for_model, contam=0.01):
        """Detects anomalies using IsolationForest on the specified features."""
        
        if not features_for_model:
            raise ValueError("No valid features available for anomaly detection. Please check your data columns.")

        missing_features = [f for f in features_for_model if f not in df_input.columns]
        if missing_features:
            raise ValueError(f"Missing features required for anomaly detection: {missing_features}. This should not happen if feature_engineer_fraud_data works correctly.")
            
        # Ensure all features for model are numeric (already done in feature_engineer_fraud_data, but as a safeguard)
        for col in features_for_model:
            if not pd.api.types.is_numeric_dtype(df_input[col]):
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0)
                if st_object: st_object.warning(f"Coerced non-numeric feature '{col}' to numeric (filled NaNs with 0) during anomaly detection step.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_input[features_for_model])

        model = IsolationForest(n_estimators=100, contamination=contam, random_state=42)
        
        df_input.loc[:, 'anomaly'] = model.fit_predict(X_scaled)
        df_input.loc[:, 'is_anomaly'] = df_input['anomaly'].apply(lambda x: 1 if x == -1 else 0)

        anomalies_df = df_input[df_input['is_anomaly'] == 1].copy()
        
        return df_input, anomalies_df, features_for_model


    def summarize_anomalies(anomalies_df, dc_name): # Removed st_object from here, pass through main func
        """Summarizes detected anomalies, optionally by date."""
        summary = []
        
        # Check if the date column exists AND is a datetime type and not all NaT
        if dc_name in anomalies_df.columns and pd.api.types.is_datetime64_any_dtype(anomalies_df[dc_name]) and not anomalies_df[dc_name].isnull().all():
            anomalies_df_copy = anomalies_df.copy() # Work on a copy
            anomalies_df_copy["Year"] = anomalies_df_copy[dc_name].dt.year
            anomalies_df_copy["Month"] = anomalies_df_copy[dc_name].dt.month_name()
            grouped = anomalies_df_copy.groupby(["Year", "Month"]).size().reset_index(name="Count")

            if not grouped.empty:
                summary.append("Anomalies by Month:")
                for _, row in grouped.iterrows():
                    summary.append(f"- {int(row['Count'])} anomalies in {row['Month']} {int(row['Year'])}")
            else:
                summary.append("No anomalies detected by month.")
        else:
            summary.append(f"Total anomalies detected: {len(anomalies_df)}")
            # if len(anomalies_df) > 0 and st_object: st_object.warning(f"Could not summarize anomalies by month. Please ensure a valid date column ('{dc_name}') is present and in datetime format.")

        return summary


    def top_anomalies(anomalies_df, value_col_hint=['TransactionAmount', 'amount', 'value', 'transaction', 'balance', 'sales']): # Removed st_object from here, pass through main func
        """Identifies and returns top anomalies based on a value column hint."""
        amount_col = None
        df_cols_lower = {col.lower(): col for col in anomalies_df.columns}

        for hint in value_col_hint:
            hint_lower = hint.lower()
            if hint_lower in df_cols_lower:
                amount_col = df_cols_lower[hint_lower]
                break
            for col_lower, original_col in df_cols_lower.items():
                if hint_lower in col_lower:
                    amount_col = original_col
                    break
            if amount_col:
                break

        if amount_col and pd.api.types.is_numeric_dtype(anomalies_df[amount_col]) and not anomalies_df.empty:
            return anomalies_df.sort_values(by=amount_col, ascending=False).head(5), amount_col # Top 5
        
        # if st_object: st_object.warning(f"Could not find a suitable numeric value column (e.g., 'TransactionAmount', 'amount') for sorting top anomalies.")
        return None, None

    # --- Plotting Functions ---
    def get_base64_image(plt_figure):
        """Converts a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        plt_figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(plt_figure) # Close the figure to free memory
        return img_base64

    def plot_anomaly_count(df):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='is_anomaly', data=df, palette=['green', 'red'], ax=ax)
        ax.set_title("Fraud vs Non-Fraud Predictions")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Fraud', 'Fraud'])
        ax.set_ylabel("Number of Transactions")
        return fig

    def plot_fraud_by_transaction_type(df):
        if 'TransactionType' not in df.columns:
            if st_object: st_object.warning("Cannot plot 'Fraud by Transaction Type': 'TransactionType' column not found.")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fraud_by_type = df.groupby(['TransactionType', 'is_anomaly']).size().unstack(fill_value=0)
        fraud_by_type.plot(kind='bar', stacked=True, color=['green', 'red'], ax=ax)
        ax.set_title("Fraud by Transaction Type")
        ax.set_ylabel("Number of Transactions")
        ax.set_xlabel("Transaction Type (Encoded)") # LabelEncoder makes them numbers
        ax.legend(["Not Fraud", "Fraud"])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def plot_fraud_over_time(df, date_col_name):
        if date_col_name not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col_name]) or df[date_col_name].isnull().all():
            if st_object: st_object.warning(f"Cannot plot 'Fraud Over Time': Invalid or missing date column '{date_col_name}'.")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 5))
        fraud_daily = df[df['is_anomaly'] == 1].groupby(df[date_col_name].dt.date).size()
        if not fraud_daily.empty:
            fraud_daily.plot(kind='line', marker='o', color='red', ax=ax)
            ax.set_title("Fraud Predictions Over Time")
            ax.set_ylabel("Fraudulent Transactions")
            ax.set_xlabel("Date")
            ax.grid(True)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        else:
            if st_object: st_object.info("No anomalies detected to plot 'Fraud Predictions Over Time'.")
            return None

    def plot_top_fraudulent_accounts(df):
        if 'AccountID' not in df.columns:
            if st_object: st_object.warning("Cannot plot 'Top Fraudulent Accounts': 'AccountID' column not found.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        top_accounts = df[df['is_anomaly'] == 1]['AccountID'].value_counts().head(10)
        
        if not top_accounts.empty:
            top_accounts.plot(kind='barh', color='darkred', ax=ax)
            ax.set_title("Top 10 Fraudulent Accounts")
            ax.set_xlabel("Number of Fraudulent Transactions")
            ax.invert_yaxis() # Highest count at the top
            plt.tight_layout()
            return fig
        else:
            if st_object: st_object.info("No anomalies detected to plot 'Top Fraudulent Accounts'.")
            return None

    def plot_correlation_heatmap(df, features_used):
        # Exclude 'anomaly' and 'is_anomaly' for correlation calculation if they exist, and add them back if desired for visualization
        cols_for_corr = [f for f in features_used if f in df.columns]
        
        if 'is_anomaly' in df.columns and 'is_anomaly' not in cols_for_corr:
            cols_for_corr.append('is_anomaly') # Ensure is_anomaly is included for correlation
        
        # Only select numeric columns for correlation
        numeric_df = df[cols_for_corr].select_dtypes(include=np.number)

        if numeric_df.empty:
            if st_object: st_object.warning("No numerical features available for correlation heatmap.")
            return None

        corr = numeric_df.corr()
        
        if 'is_anomaly' in corr.columns:
            sorted_corr = corr[['is_anomaly']].sort_values(by='is_anomaly', ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, max(6, len(sorted_corr) * 0.5))) # Adjust size based on number of features
            sns.heatmap(sorted_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True)
            ax.set_title("Correlation of Features with Fraud Prediction")
            plt.tight_layout()
            return fig
        else:
            if st_object: st_object.warning("'is_anomaly' column not found for correlation heatmap.")
            return None

    # Main execution flow for fraud_detection_analysis
    plot_images = {} # Dictionary to store base64 encoded plot images

    try:
        if st_object: st_object.info("Loading data...")
        df_raw = load_data(file_path)
    except Exception as e:
        raise ValueError(f"Data loading failed for fraud detection: {e}")

    try:
        if st_object: st_object.info("Engineering features for fraud detection...")
        df_featured, features_for_model = feature_engineer_fraud_data(df_raw, primary_date_col=date_col_name)
    except Exception as e:
        raise ValueError(f"Feature engineering failed for fraud detection: {e}")

    try:
        if st_object: st_object.info("Detecting anomalies using IsolationForest...")
        df_with_anomalies, anomalies_df, used_features = detect_anomalies(df_featured.copy(), features_for_model, contam=contamination)
    except ValueError as e:
        raise ValueError(f"Anomaly detection failed: {e}")
    except RuntimeError as e:
        raise RuntimeError(f"An unexpected error occurred during anomaly detection: {e}")
    except Exception as e:
        raise RuntimeError(f"A general error occurred during anomaly detection: {e}")


    anomaly_summary_list = summarize_anomalies(anomalies_df, dc_name=date_col_name)
    top_anomalies_df, amount_col_identified = top_anomalies(anomalies_df)

    # --- Generate Plots ---
    if st_object: st_object.info("Generating visualizations...")
    
    # Plot 1: Anomaly Count Plot
    fig_count = plot_anomaly_count(df_with_anomalies)
    if fig_count: plot_images['anomaly_count'] = get_base64_image(fig_count)
    
    # Plot 2: Fraud by Transaction Type
    fig_type = plot_fraud_by_transaction_type(df_with_anomalies)
    if fig_type: plot_images['fraud_by_type'] = get_base64_image(fig_type)

    # Plot 3: Fraud Over Time (requires valid date column)
    fig_time = plot_fraud_over_time(df_with_anomalies, date_col_name)
    if fig_time: plot_images['fraud_over_time'] = get_base64_image(fig_time)

    # Plot 4: Top Fraudulent Accounts (requires 'AccountID')
    fig_accounts = plot_top_fraudulent_accounts(df_with_anomalies)
    if fig_accounts: plot_images['top_fraud_accounts'] = get_base64_image(fig_accounts)

    # Plot 5: Correlation Heatmap
    fig_corr = plot_correlation_heatmap(df_with_anomalies, used_features)
    if fig_corr: plot_images['correlation_heatmap'] = get_base64_image(fig_corr)


    return df_with_anomalies, anomalies_df, anomaly_summary_list, top_anomalies_df, amount_col_identified, plot_images