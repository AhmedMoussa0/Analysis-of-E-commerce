import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(page_title="📊 E-Commerce Analytics Dashboard", layout="wide")
st.title("📊 E-Commerce Analytics Dashboard")

# Load and prepare data
@st.cache_data
def load_data():
  df = pd.read_csv("ecommerce_customer_data_large.csv")
  df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
  df['Year'] = df['Purchase Date'].dt.year
  df['Month'] = df['Purchase Date'].dt.month
  df['Quarter'] = df['Purchase Date'].dt.quarter
  df['Day'] = df['Purchase Date'].dt.day
  df['Returns'] = df['Returns'].fillna(0)
  
  # Create age groups
  age_bins = [0, 20, 30, 40, 50, 60, 100]
  age_labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60+']
  df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
  
  return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
categories = st.sidebar.multiselect(
  "Product Categories",
  options=df["Product Category"].unique(),
  default=df["Product Category"].unique()
)

years = st.sidebar.multiselect(
  "Years",
  options=sorted(df["Year"].unique()),
  default=sorted(df["Year"].unique())
)

genders = st.sidebar.multiselect(
  "Genders",
  options=df["Gender"].unique(),
  default=df["Gender"].unique()
)

# Apply filters
filtered_df = df[
  (df["Product Category"].isin(categories)) &
  (df["Year"].isin(years)) &
  (df["Gender"].isin(genders))
]

# KPI Cards
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${filtered_df['Total Purchase Amount'].sum():,.2f}")
col2.metric("Avg Order Value", f"${filtered_df['Total Purchase Amount'].mean():,.2f}")
col3.metric("Total Customers", filtered_df['Customer ID'].nunique())
col4.metric("Churn Rate", f"{filtered_df['Churn'].mean() * 100:.2f}%")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Overview", "Customer Analytics", "Business Performance"])

with tab1:
  st.subheader("📋 Data Overview")
  
  # Descriptive statistics
  st.write("### Descriptive Statistics")
  
  col1, col2 = st.columns(2)
  
  with col1:
      st.write("**Numerical Columns**")
      st.dataframe(filtered_df.select_dtypes(include=np.number).describe())
  
  with col2:
      st.write("**Categorical Columns**")
      st.dataframe(filtered_df.select_dtypes(exclude=np.number).describe())
  
  # Data sample
  st.write("### Data Sample")
  st.dataframe(filtered_df.head())

with tab2:
  st.subheader("👥 Customer Demographics Analysis")
  
  # Age and gender distribution
  st.write("### Customer Distribution by Age and Gender")
  col1, col2 = st.columns(2)
  
  with col1:
      fig = px.histogram(filtered_df, x='Age Group', color='Gender', barmode='group',
                        title='Customer Count by Age Group and Gender')
      st.plotly_chart(fig, use_container_width=True)
  with col2:
      gender_dist = filtered_df.groupby('Gender').size().reset_index(name='Count')
      fig = px.pie(gender_dist, values='Count', names='Gender',
                  title='Gender Distribution')
      st.plotly_chart(fig, use_container_width=True)
  
  # Spending patterns
  st.write("### Spending Patterns")
  avg_spending = filtered_df.groupby(['Age Group', 'Gender'])['Total Purchase Amount'].mean().reset_index()
  fig = px.bar(avg_spending, x='Age Group', y='Total Purchase Amount', color='Gender',
              barmode='group', title='Average Spending by Age Group and Gender')
  st.plotly_chart(fig, use_container_width=True)
  
  # Payment method preferences
  st.write("### Payment Method Preferences")
  payment_pref = filtered_df.groupby(['Age Group', 'Gender', 'Payment Method']).size().reset_index(name='Count')
  fig = px.bar(payment_pref, x='Age Group', y='Count', color='Payment Method',
              facet_col='Gender', title='Payment Method Preferences by Age and Gender')
  st.plotly_chart(fig, use_container_width=True)

with tab3:
  st.subheader("📊 Business Performance Analysis")
  
  # Sales trends
  st.write("### Sales Trends Over Time")
  time_period = st.radio("Time Period", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
  
  if time_period == "Monthly":
      sales_trend = filtered_df.groupby(['Year', 'Month'])['Total Purchase Amount'].sum().reset_index()
      sales_trend['Date'] = pd.to_datetime(sales_trend[['Year', 'Month']].assign(day=1))
      fig = px.line(sales_trend, x='Date', y='Total Purchase Amount', title='Monthly Sales Trend')
  elif time_period == "Quarterly":
      sales_trend = filtered_df.groupby(['Year', 'Quarter'])['Total Purchase Amount'].sum().reset_index()
      sales_trend['Date'] = sales_trend['Year'].astype(str) + 'Q' + sales_trend['Quarter'].astype(str)
      fig = px.line(sales_trend, x='Date', y='Total Purchase Amount', title='Quarterly Sales Trend')
  else:
      sales_trend = filtered_df.groupby('Year')['Total Purchase Amount'].sum().reset_index()
      fig = px.line(sales_trend, x='Year', y='Total Purchase Amount', title='Yearly Sales Trend')
  
  st.plotly_chart(fig, use_container_width=True)
  
  # Product category performance
  st.write("### Product Category Performance")
  col1, col2 = st.columns(2)
  
  with col1:
      cat_revenue = filtered_df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
      fig = px.pie(cat_revenue, values='Total Purchase Amount', names='Product Category',
                  title='Revenue by Product Category')
      st.plotly_chart(fig, use_container_width=True)
  
  with col2:
      cat_sales = filtered_df.groupby('Product Category')['Quantity'].sum().reset_index()
      fig = px.bar(cat_sales, x='Product Category', y='Quantity',
                  title='Units Sold by Product Category')
      st.plotly_chart(fig, use_container_width=True)
  
  # Return analysis
  st.write("### Return Analysis")
  col1, col2 = st.columns(2)
  
  with col1:
      return_rate = filtered_df.groupby('Product Category')['Returns'].mean().reset_index()
      fig = px.bar(return_rate, x='Product Category', y='Returns',
                  title='Return Rate by Product Category')
      st.plotly_chart(fig, use_container_width=True)
  
  with col2:
      return_payment = filtered_df.groupby('Payment Method')['Returns'].mean().reset_index()
      fig = px.bar(return_payment, x='Payment Method', y='Returns',
                  title='Return Rate by Payment Method')
      st.plotly_chart(fig, use_container_width=True)
  
  # Customer purchase behavior
  st.write("### Customer Purchase Behavior")
  purchase_freq = filtered_df.groupby('Customer ID').size().reset_index(name='Purchase Count')
  st.bar_chart(purchase_freq)
  fig = px.histogram(purchase_freq, x='Purchase Count', title='Purchase Frequency Distribution')
  st.plotly_chart(fig, use_container_width=True)

# Download options
st.sidebar.header("📥 Export Options")
if st.sidebar.button("Download Filtered Data as CSV"):
  csv = filtered_df.to_csv(index=False).encode('utf-8')
  st.sidebar.download_button(
      label="Download CSV",
      data=csv,
      file_name="filtered_ecommerce_data.csv",
      mime="text/csv"
  )

