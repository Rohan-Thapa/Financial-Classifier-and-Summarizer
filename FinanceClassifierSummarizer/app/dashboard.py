# The sample dashboard as for the test and for checking whether the model is working or not
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from inference import TransactionClassifier
from config import config
import torch
from summarizer import simple_summarizer
from transactions_type import classify_df

# Adding the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initializing the classifier
classifier = TransactionClassifier()


def main():
    st.set_page_config(
        page_title="Classify'n'Summarize | Financial Intelligence",
        page_icon="💰",
        layout="wide"
    )

    # ADD: Device warning if using CPU
    if torch.cuda.is_available():
        st.sidebar.success("✅ Using GPU acceleration")
        st.sidebar.text("The AI is using GPU and is fast.")
    else:
        st.sidebar.warning("⚠️ Using CPU - performance may be slower")
        st.sidebar.text("The AI is using CPU due to CUDA unavailable and is 3-5x slower.")

    st.sidebar.divider()
    st.sidebar.text("Created by Summarizers Group")

    # Custom CSS for the dashboard
    st.markdown("""
    <style>
        .header {
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .metric-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }
        .stProgress > div > div > div > div {
            background-color: #2575fc;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header content for the dashboard
    st.markdown("""
    <div class="header">
        <h1>💰 Classify’n’Summarize</h1>
        <p>AI-powered Transaction Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame(columns=["Transaction", "Category", "Amount", "Confidence"])

    # Input Section for the data input
    with st.expander("💳 Input Transactions", expanded=True):
        input_method = st.radio("Input method:", ["Single Transaction", "Batch Input"])
        transactions = []

        if input_method == "Single Transaction":
            tx = st.text_input("Transaction:", "Paid NPR 350 for NTC mobile topup")
            transactions = [tx]
        else:
            txs = st.text_area("Transactions (one per line)",
                               "Paid NPR 350 for NTC mobile topup\nSent Rs 2000 to Sita Bank for tuition")
            transactions = [tx.strip() for tx in txs.split("\n") if tx.strip()]

        if st.button("Analyze Transactions", type="primary") and transactions:
            with st.spinner("Processing..."):
                results = classifier.categorize(transactions)

                # Converting to the DataFrame
                new_data = []
                for r in results:
                    new_data.append({
                        "Transaction": r["transaction"],
                        "Category": r["category"],
                        "Amount": r["amount"],
                        "Confidence": f"{r['confidence']:.1%}"
                    })

                new_df = pd.DataFrame(new_data)
                st.session_state.results = pd.concat([st.session_state.results, new_df])
                st.success(f"✅ Processed {len(transactions)} transactions")

    # Results Display after prediction
    if not st.session_state.results.empty:
        st.subheader("📊 Analysis Results")

        # Data Table
        st.dataframe(st.session_state.results, use_container_width=True)

        # Visualizations of the data
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Category Distribution")
            fig1 = px.pie(
                st.session_state.results,
                names='Category',
                color='Category',
                color_discrete_map=config.CATEGORY_COLORS
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Spending by Category")
            agg_df = st.session_state.results.groupby('Category')['Amount'].sum().reset_index()
            fig2 = px.bar(
                agg_df,
                x='Category',
                y='Amount',
                color='Category',
                color_discrete_map=config.CATEGORY_COLORS
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Summarization of the transactions")
        if st.button("Summarize the Transactions", type="primary"):
            with st.expander("This is the summarization text of the transactions.", expanded=True):
                st.write(simple_summarizer(st.session_state.results)) # Here the summarized text should be

                type_data = classify_df(st.session_state.results)

                expense_df = type_data[
                    type_data['Type'] == 'Expense'].copy()  # getting the data related to the expenses
                income_df = type_data[
                    type_data['Type'] == 'Income'].copy()  # getting the incomes also for the proper management

                # generating the expense and income summary
                expense_summary = expense_df.groupby('Category', as_index=False)['Amount'].sum()
                income_summary = income_df.groupby('Category', as_index=False)['Amount'].sum()
                total_income = income_summary['Amount'].sum()

                # Calculating the total on the expenses
                total_expense = expense_summary['Amount'].sum()
                expense_summary['Percentage'] = round((expense_summary['Amount'] / total_expense) * 100, 1)

                st.write("Expenses Summary:")
                st.dataframe(expense_summary, use_container_width=True)

                st.write(f"Total Income: NPR {total_income}")
                st.write(f"Total Expense: NPR {total_expense}")
                st.write(f"Net Balance: NPR {total_income - total_expense}")

                st.subheader("Income vs Expenses Chart")
                summary = type_data.groupby('Type', as_index=False)['Amount'].sum()
                # Ensuring the consistent ordering
                summary = summary.set_index('Type').reindex(['Income', 'Expense']).reset_index()

                fig = px.bar(
                    summary,
                    x="Type",
                    y="Amount",
                    title="Total Income and Expense",
                    text="Amount"
                )
                fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                fig.update_layout(yaxis_title="Amount (NPR)", xaxis_title="Transaction Type")

                # Displaying the chart for better understanding.
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
# The sample dashboard for the project.