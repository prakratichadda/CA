# YOUR AI CHARTERED ACCOUNTANT

## Overview

The **YOUR AI CHARTERED ACCOUNTANT** is a powerful, interactive web application built with Streamlit designed to empower individuals and businesses with various financial analytical capabilities. From forecasting future trends and detecting anomalies to managing invoices and understanding tax liabilities, this suite provides intuitive tools for data-driven financial decision-making.

The application leverages Python's robust data science ecosystem, including `pandas` for data manipulation, `scikit-learn` for machine learning models (like anomaly detection), and `plotly` for interactive visualizations.

## Features

This suite offers four core functionalities, accessible via the sidebar navigation:

### 1. 📈 Financial Forecasting

* **Purpose:** Predicts future financial trends (e.g., sales, revenue, demand) and identifies unusual data points (anomalies) in historical time-series data.
* **Input:** CSV file with historical numerical data and an optional date column.
* **Outputs:** Interactive plots showing forecasts and anomalies, tabular data of forecasted values, and a list of detected anomalies. Includes visualizations for numeric trends, sales vs. target, market indicators, and feature correlation.

### 2. 🕵️‍♂️ Fraud Detection

* **Purpose:** Identifies potentially fraudulent or anomalous transactions within your operational or financial datasets using Isolation Forest (an unsupervised machine learning algorithm).
* **Input:** CSV file containing transactional data (numerical and categorical).
* **Outputs:** Visualizations of fraud counts, fraud over time, top fraudulent accounts, and correlation with features. Provides a list of detected anomalies and a summary of suspicious activities.

### 3. 💰 Tax Compliance Calculator (India)

* **Purpose:** A simplified calculator for estimating individual income tax liability in India based on gross income, eligible deductions, and the chosen tax year (primarily for the Old Tax Regime).
* **Input:** User-entered gross annual income, eligible deductions, and tax year via numeric inputs.
* **Outputs:** Detailed breakdown of taxable income, tax before cess, health & education cess, and total tax liability. Includes a visual representation of tax contribution per slab.

### 4. 🧾 Invoice Processing & Analysis

* **Purpose:** Provides a comprehensive analysis of invoice data, including customer segmentation, further fraud detection based on business rules, entity extraction, and budget vs. actuals comparison.
* **Input:** CSV file with detailed invoice line-item data.
* **Outputs:**
    * **Customer Segmentation:** Top customer segments by revenue, revenue trends by city and month.
    * **Invoice Fraud Detection:** Identifies suspicious invoices based on custom rules (e.g., high value, duplicates).
    * **Extracted Entities:** Summarizes key details like names, emails, and product IDs.
    * **Budget vs. Actuals:** Compares actual invoice amounts against simulated budgets, particularly by job role.
    * **Audit Flags:** Highlights invoices for further manual review.

## Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning algorithms (e.g., `IsolationForest`).
* **Plotly Express / Graph Objects:** For interactive and compelling data visualizations.
* **Matplotlib:** For static plots in some sections.
* **base64:** For encoding/decoding images for display in Streamlit.

## Getting Started

Follow these steps to set up and run the Financial Analysis Suite on your local machine.

### Prerequisites

* Python 3.8 or higher
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name # Navigate into your project directory
    ```
    (Replace `your-username/your-repository-name.git` with the actual URL of your GitHub repository.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` yet, create one by running `pip freeze > requirements.txt` after installing all necessary libraries (streamlit, pandas, numpy, scikit-learn, plotly, matplotlib).*

### Running the Application

1.  **Ensure your virtual environment is active.**
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

    This command will open the application in your default web browser (usually at `http://localhost:8501`).

## CSV Data Requirements

To ensure the features work correctly, please prepare your CSV files according to these guidelines:

### 📈 Financial Forecasting

* **Mandatory:** At least one numerical column representing the time-series data to be forecasted (e.g., `'sales'`, `'revenue'`, `'demand'`). You will specify its name as the 'Target Column' in the sidebar.
* **Highly Recommended:** A 'Date' column in a recognizable format (e.g., `'YYYY-MM-DD'`, `'MM/DD/YYYY'`). This helps in plotting and time-series analysis. If missing or invalid, a simple time step index will be used.
* **Optional:** Other numerical columns (e.g., `'gdp_growth'`, `'unemployment_rate'`, `'inflation_rate'`, `'marketing_spend'`) for correlation analysis and additional visualizations.
* **Structure:** Each row should represent a sequential time period (e.g., daily, weekly, monthly).

### 🕵️‍♂️ Fraud Detection

* **Mandatory:** At least one numerical column (e.g., `'TransactionAmount'`, `'LoginAttempts'`, `'AccountBalance'`) for the anomaly detection model to analyze.
* **Highly Recommended:** A primary `Date` column (e.g., `'TransactionDate'`, `'ActivityDate'`) for time-based feature engineering and summaries. You can specify its name in the sidebar.
* **Optional but Recommended:** Other relevant numerical and categorical columns that might contain patterns of normal/fraudulent behavior, such as:
    * **Numerical:** `'CustomerAge'`, `'NumberOfItems'`, `'IPAddress'` (if converted to numerical features), `'TimeSinceLastTransaction'`.
    * **Categorical (will be one-hot encoded or processed):** `'TransactionType'`, `'Location'`, `'Channel'`, `'MerchantID'`, `'CustomerID'`, `'ProductCategory'`.
* **Structure:** Each row should represent a single event or transaction.

### 🧾 Invoice Processing & Analysis

* **Mandatory columns:**
    * `invoice_date` (Date of the invoice, e.g., `'YYYY-MM-DD'`)
    * `amount` (Numerical value of the invoice item)
    * `product_id` (Identifier for the product/service)
* **Recommended/Optional columns for full analysis:**
    * `invoice_id` (Unique ID for the invoice)
    * `first_name`, `last_name` (Client's name)
    * `email` (Client's email address)
    * `city` (Client's city for geographical segmentation)
    * `job` (Client's job role for occupational segmentation and budget analysis)
    * `qty` (Quantity of the product/service)
    * `product_name`, `category` (Additional product details)
* **Structure:** Each row should ideally represent a single line item on an invoice.

## Demo Data (Included)

This repository includes small, illustrative CSV files in the project root to help you test the features immediately:

* `invoices.csv` (for Invoice Processing)
* `simulated_financial_forecasting_data.csv` (for Financial Forecasting)
* `bank_transactions_data_2.csv` (for Fraud Detection)
* `tax_risk_dataset.csv` (for potential future use or if integrated with tax compliance)

Feel free to use these or upload your own compatible data.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, feel free to reach out:

* **Your Name:** prakratichadda
* **Email:** chaddaprakrati@gmail.com

---
