import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Directory containing this script (assumes CSVs are in the same folder)
BASE_DIR = Path(__file__).parent

# Filenames (adjust if your files are in a different folder)
FILES = {
	"customer_feedback": BASE_DIR / "Customer_Feedback_Data.csv",
	"product_offering": BASE_DIR / "Product_Offering_Data.csv",
	"transaction": BASE_DIR / "Transaction_Data.csv",
}

def load_csv(path: Path):
	if not path.exists():
		print(f"Warning: {path} not found.")
		return None
	return pd.read_csv(path)


def clean_customer_feedback(df):
	"""Data cleaning for customer feedback."""
	if df is None or df.empty:
		return df
	
	print("\n--- Cleaning Customer Feedback ---")
	initial_rows = len(df)
	
	# Remove duplicates
	df = df.drop_duplicates(subset=['Customer_ID']).copy()
	print(f"Removed duplicates: {initial_rows - len(df)} rows")
	
	# Fill missing Satisfaction_Score with median
	if 'Satisfaction_Score' in df.columns:
		median_satisfaction = df['Satisfaction_Score'].median()
		df['Satisfaction_Score'] = df['Satisfaction_Score'].fillna(median_satisfaction)
		print(f"Imputed Satisfaction_Score with median: {median_satisfaction}")
	
	# Fill missing Feedback_Comments with "No comment"
	if 'Feedback_Comments' in df.columns:
		df['Feedback_Comments'] = df['Feedback_Comments'].fillna('No comment')
	
	# Remove outliers in Likelihood_to_Recommend (outside 0-10 range)
	if 'Likelihood_to_Recommend' in df.columns:
		df = df[(df['Likelihood_to_Recommend'] >= 0) & (df['Likelihood_to_Recommend'] <= 10)]
	
	print(f"Final rows after cleaning: {len(df)}")
	return df


def clean_transaction(df):
	"""Data cleaning for transaction data."""
	if df is None or df.empty:
		return df
	
	print("\n--- Cleaning Transaction Data ---")
	initial_rows = len(df)
	
	# Remove duplicates on Transaction_ID
	df = df.drop_duplicates(subset=['Transaction_ID']).copy()
	print(f"Removed duplicates: {initial_rows - len(df)} rows")
	
	# Fill missing Transaction_Amount with median
	if 'Transaction_Amount' in df.columns:
		median_amount = df['Transaction_Amount'].median()
		df['Transaction_Amount'] = df['Transaction_Amount'].fillna(median_amount)
		print(f"Imputed Transaction_Amount with median: {median_amount}")
	
	# Remove rows with missing Transaction_Type
	if 'Transaction_Type' in df.columns:
		df = df.dropna(subset=['Transaction_Type'])
	
	# Remove rows with missing Transaction_Date
	if 'Transaction_Date' in df.columns:
		df = df.dropna(subset=['Transaction_Date'])
	
	# Remove outliers: transaction amounts > 10,000 (potentially erroneous)
	if 'Transaction_Amount' in df.columns:
		outliers_removed = len(df[df['Transaction_Amount'] > 10000])
		df = df[df['Transaction_Amount'] <= 10000]
		if outliers_removed > 0:
			print(f"Removed {outliers_removed} outlier transactions (amount > 10000)")
	
	print(f"Final rows after cleaning: {len(df)}")
	return df


def clean_product_offering(df):
	"""Data cleaning for product offering."""
	if df is None or df.empty:
		return df
	
	print("\n--- Cleaning Product Offering Data ---")
	initial_rows = len(df)
	
	# Remove duplicates on Product_ID
	df = df.drop_duplicates(subset=['Product_ID']).copy()
	print(f"Removed duplicates: {initial_rows - len(df)} rows")
	
	# Convert Target_Age_Group to object type first, then fill missing
	if 'Target_Age_Group' in df.columns:
		df['Target_Age_Group'] = df['Target_Age_Group'].astype('object')
		df['Target_Age_Group'] = df['Target_Age_Group'].fillna('Unknown')
	
	# Fill missing Target_Income_Group
	if 'Target_Income_Group' in df.columns:
		df['Target_Income_Group'] = df['Target_Income_Group'].fillna('Medium')
	
	# Remove rows with missing critical fields
	df = df.dropna(subset=['Product_ID', 'Product_Name', 'Product_Type', 'Risk_Level'])
	
	print(f"Final rows after cleaning: {len(df)}")
	return df


def engineer_customer_feedback(df):
	"""Feature engineering for customer feedback data."""
	if df is None or df.empty:
		return df
	
	# Create satisfaction category
	df['Satisfaction_Category'] = pd.cut(
		df['Satisfaction_Score'],
		bins=[0, 3.5, 7, 10],
		labels=['Low', 'Medium', 'High'],
		include_lowest=True
	)
	
	# Comment length feature
	df['Comment_Length'] = df['Feedback_Comments'].fillna('').str.len()
	
	# Satisfaction-to-Recommendation gap
	df['Satisfaction_Recommendation_Gap'] = (
		df['Satisfaction_Score'].fillna(0) - df['Likelihood_to_Recommend']
	).abs()
	
	return df


def engineer_transaction(df):
	"""Feature engineering for transaction data."""
	if df is None or df.empty:
		return df
	
	# Parse transaction date
	df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
	df['Transaction_Year'] = df['Transaction_Date'].dt.year
	df['Transaction_Month'] = df['Transaction_Date'].dt.month
	df['Transaction_Day'] = df['Transaction_Date'].dt.day
	df['Transaction_DayOfWeek'] = df['Transaction_Date'].dt.dayofweek
	
	# Create amount category
	df['Amount_Category'] = pd.cut(
		df['Transaction_Amount'],
		bins=[0, 500, 1500, 5000, np.inf],
		labels=['Small', 'Medium', 'Large', 'VeryLarge'],
		include_lowest=True
	)
	
	# Aggregate features per customer
	customer_agg = df.groupby('Customer_ID').agg({
		'Transaction_Amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
		'Transaction_ID': 'count'
	}).fillna(0)
	
	# Flatten column names
	customer_agg.columns = [
		'Transaction_Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Min_Amount', 'Max_Amount',
		'Total_Transactions'
	]
	customer_agg = customer_agg.drop('Total_Transactions', axis=1)
	
	print("\n=== Transaction Aggregates (Sample) ===")
	print(customer_agg.head())
	
	return df, customer_agg


def engineer_product_offering(df):
	"""Feature engineering for product offering data."""
	if df is None or df.empty:
		return df
	
	# Encode categorical features
	df['Risk_Level_Encoded'] = pd.factorize(df['Risk_Level'])[0]
	df['Income_Numeric'] = df['Target_Income_Group'].map({'Low': 1, 'Medium': 2, 'High': 3})
	
	return df


def visualize_customer_feedback(df, output_dir):
	"""Generate visualizations for customer feedback data."""
	if df is None or df.empty:
		return
	
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle('Customer Feedback Analysis', fontsize=16, fontweight='bold')
	
	# Satisfaction score distribution
	df['Satisfaction_Score'].dropna().hist(bins=15, ax=axes[0, 0], color='skyblue', edgecolor='black')
	axes[0, 0].set_title('Satisfaction Score Distribution')
	axes[0, 0].set_xlabel('Satisfaction Score')
	axes[0, 0].set_ylabel('Frequency')
	
	# Likelihood to recommend distribution
	df['Likelihood_to_Recommend'].hist(bins=15, ax=axes[0, 1], color='lightcoral', edgecolor='black')
	axes[0, 1].set_title('Likelihood to Recommend Distribution')
	axes[0, 1].set_xlabel('Likelihood Score')
	axes[0, 1].set_ylabel('Frequency')
	
	# Satisfaction vs Recommendation scatter
	axes[1, 0].scatter(df['Satisfaction_Score'], df['Likelihood_to_Recommend'], alpha=0.6, color='green')
	axes[1, 0].set_title('Satisfaction vs Likelihood to Recommend')
	axes[1, 0].set_xlabel('Satisfaction Score')
	axes[1, 0].set_ylabel('Likelihood Score')
	
	# Satisfaction category breakdown
	if 'Satisfaction_Category' in df.columns:
		df['Satisfaction_Category'].value_counts().plot(kind='bar', ax=axes[1, 1], color=['#ff9999', '#ffcc99', '#99ccff'])
		axes[1, 1].set_title('Satisfaction Category Breakdown')
		axes[1, 1].set_xlabel('Category')
		axes[1, 1].set_ylabel('Count')
		axes[1, 1].tick_params(axis='x', rotation=45)
	
	plt.tight_layout()
	filepath = output_dir / "01_Customer_Feedback_Analysis.png"
	plt.savefig(filepath, dpi=300, bbox_inches='tight')
	print(f"✓ Saved visualization: {filepath.name}")
	plt.close()


def visualize_transactions(df, trans_agg, output_dir):
	"""Generate visualizations for transaction data."""
	if df is None or df.empty:
		return
	
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle('Transaction Analysis', fontsize=16, fontweight='bold')
	
	# Transaction amount distribution
	df['Transaction_Amount'].dropna().hist(bins=30, ax=axes[0, 0], color='skyblue', edgecolor='black')
	axes[0, 0].set_title('Transaction Amount Distribution')
	axes[0, 0].set_xlabel('Amount')
	axes[0, 0].set_ylabel('Frequency')
	
	# Transaction type breakdown
	if 'Transaction_Type' in df.columns:
		df['Transaction_Type'].value_counts().plot(kind='bar', ax=axes[0, 1], color='coral')
		axes[0, 1].set_title('Transaction Type Breakdown')
		axes[0, 1].set_xlabel('Type')
		axes[0, 1].set_ylabel('Count')
		axes[0, 1].tick_params(axis='x', rotation=45)
	
	# Amount category breakdown
	if 'Amount_Category' in df.columns:
		df['Amount_Category'].value_counts().plot(kind='bar', ax=axes[1, 0], color=['#99ff99', '#ffff99', '#ff9999', '#ff6666'])
		axes[1, 0].set_title('Amount Category Breakdown')
		axes[1, 0].set_xlabel('Category')
		axes[1, 0].set_ylabel('Count')
		axes[1, 0].tick_params(axis='x', rotation=45)
	
	# Top 10 customers by transaction count
	if not trans_agg.empty:
		top_customers = trans_agg.nlargest(10, 'Transaction_Count')
		top_customers['Transaction_Count'].plot(kind='barh', ax=axes[1, 1], color='purple')
		axes[1, 1].set_title('Top 10 Customers by Transaction Count')
		axes[1, 1].set_xlabel('Transaction Count')
	
	plt.tight_layout()
	filepath = output_dir / "02_Transaction_Analysis.png"
	plt.savefig(filepath, dpi=300, bbox_inches='tight')
	print(f"✓ Saved visualization: {filepath.name}")
	plt.close()


def visualize_products(df, output_dir):
	"""Generate visualizations for product offering data."""
	if df is None or df.empty:
		return
	
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle('Product Offering Analysis', fontsize=16, fontweight='bold')
	
	# Product type breakdown
	if 'Product_Type' in df.columns:
		df['Product_Type'].value_counts().plot(kind='bar', ax=axes[0], color='teal')
		axes[0].set_title('Product Type Distribution')
		axes[0].set_xlabel('Product Type')
		axes[0].set_ylabel('Count')
		axes[0].tick_params(axis='x', rotation=45)
	
	# Risk level breakdown
	if 'Risk_Level' in df.columns:
		risk_colors = {'Low': '#99ff99', 'Medium': '#ffff99', 'High': '#ff9999'}
		risk_counts = df['Risk_Level'].value_counts()
		colors = [risk_colors.get(x, 'gray') for x in risk_counts.index]
		risk_counts.plot(kind='pie', ax=axes[1], colors=colors, autopct='%1.1f%%')
		axes[1].set_title('Risk Level Distribution')
		axes[1].set_ylabel('')
	
	plt.tight_layout()
	filepath = output_dir / "03_Product_Analysis.png"
	plt.savefig(filepath, dpi=300, bbox_inches='tight')
	print(f"✓ Saved visualization: {filepath.name}")
	plt.close()


def main():
	dfs = {name: load_csv(path) for name, path in FILES.items()}

	# Print original data
	print("=" * 80)
	print("ORIGINAL DATA OVERVIEW")
	print("=" * 80)
	for name, df in dfs.items():
		print(f"\n=== {name} ===")
		if df is None:
			continue
		print(df.head())
		print("\nColumns:", list(df.columns))
		print("\nMissing values per column:\n", df.isnull().sum())
	
	# Data Cleaning
	print("\n" + "=" * 80)
	print("DATA CLEANING")
	print("=" * 80)
	dfs['customer_feedback'] = clean_customer_feedback(dfs['customer_feedback'])
	dfs['transaction'] = clean_transaction(dfs['transaction'])
	dfs['product_offering'] = clean_product_offering(dfs['product_offering'])
	
	# Print cleaned data summary
	print("\n" + "=" * 80)
	print("CLEANED DATA OVERVIEW")
	print("=" * 80)
	for name, df in dfs.items():
		if df is not None:
			print(f"\n=== {name} (cleaned) ===")
			print(f"Shape: {df.shape}")
			print(f"Missing values:\n{df.isnull().sum().sum()} total missing values")

	# Feature Engineering
	print("\n" + "=" * 80)
	print("FEATURE ENGINEERING")
	print("=" * 80)
	
	# Engineer customer feedback
	if dfs['customer_feedback'] is not None:
		dfs['customer_feedback'] = engineer_customer_feedback(dfs['customer_feedback'])
		print("\n=== Customer Feedback (Engineered) ===")
		print(dfs['customer_feedback'].head())
		print("New columns:", [c for c in dfs['customer_feedback'].columns if c not in ['Customer_ID', 'Satisfaction_Score', 'Feedback_Comments', 'Likelihood_to_Recommend']])
	
	# Engineer transaction
	if dfs['transaction'] is not None:
		dfs['transaction'], trans_agg = engineer_transaction(dfs['transaction'])
		print("\n=== Transaction (Engineered) ===")
		print(dfs['transaction'].head())
		print("New columns:", [c for c in dfs['transaction'].columns if c not in ['Transaction_ID', 'Customer_ID', 'Transaction_Date', 'Transaction_Amount', 'Transaction_Type']])
	
	# Engineer product offering
	if dfs['product_offering'] is not None:
		dfs['product_offering'] = engineer_product_offering(dfs['product_offering'])
		print("\n=== Product Offering (Engineered) ===")
		print(dfs['product_offering'].head())
		print("New columns:", [c for c in dfs['product_offering'].columns if c not in ['Product_ID', 'Product_Name', 'Product_Type', 'Risk_Level', 'Target_Age_Group', 'Target_Income_Group']])
	
	# Save engineered datasets
	print("\n" + "=" * 80)
	print("SAVING ENGINEERED DATASETS")
	print("=" * 80)
	if dfs['customer_feedback'] is not None:
		dfs['customer_feedback'].to_csv(BASE_DIR / "Customer_Feedback_Engineered.csv", index=False)
		print("✓ Saved: Customer_Feedback_Engineered.csv")
	
	if dfs['transaction'] is not None:
		dfs['transaction'].to_csv(BASE_DIR / "Transaction_Engineered.csv", index=False)
		trans_agg.to_csv(BASE_DIR / "Transaction_Aggregates.csv")
		print("✓ Saved: Transaction_Engineered.csv")
		print("✓ Saved: Transaction_Aggregates.csv")
	
	if dfs['product_offering'] is not None:
		dfs['product_offering'].to_csv(BASE_DIR / "Product_Offering_Engineered.csv", index=False)
		print("✓ Saved: Product_Offering_Engineered.csv")
	
	# Generate visualizations
	print("\n" + "=" * 80)
	print("GENERATING VISUALIZATIONS")
	print("=" * 80)
	visualize_customer_feedback(dfs['customer_feedback'], BASE_DIR)
	visualize_transactions(dfs['transaction'], trans_agg, BASE_DIR)
	visualize_products(dfs['product_offering'], BASE_DIR)
	print("\nAll visualizations saved to:", BASE_DIR)
	print("Open the PNG files to view charts.")


if __name__ == "__main__":
	main()
