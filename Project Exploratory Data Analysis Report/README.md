# DMP

Data pipeline project: loads three CSVs, cleans data, performs feature engineering, generates visualizations, and saves engineered CSVs.

Quick start

1. Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

2. Run the pipeline:

```powershell
py main.py
```

What it produces

- `Customer_Feedback_Engineered.csv`
- `Transaction_Engineered.csv`
- `Transaction_Aggregates.csv`
- `Product_Offering_Engineered.csv`
- `01_Customer_Feedback_Analysis.png`
- `02_Transaction_Analysis.png`
- `03_Product_Analysis.png`

Notes

- Large or sensitive CSV files should not be pushed to GitHub. If you want to include sample data, add small sample files only.
- To push from your machine you can use the GitHub web UI (upload) or install Git/GitHub Desktop.
