# Customer Segmentation Project

## Objective
Segment customers based on behavior and demographics to identify actionable customer groups.

## Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn, K-Means Clustering.

## Dataset
500 customer records containing demographic and behavioral variables such as age, annual income, purchase frequency, average order value, website visits, discount usage, recency and spending score.

## Methodology
1. Load and inspect the data
2. Handle missing numeric values using median imputation
3. Select demographic and behavioral features
4. Standardize features using StandardScaler
5. Test K-Means for K=2 through K=8
6. Select the best K using silhouette score
7. Assign customers to clusters
8. Visualize customer segments
9. Convert clusters into business-oriented customer profiles

## Model Result
Best K = **2**
Silhouette Score = **0.223**

## Business Recommendations
- High-Value Customers: retention, loyalty rewards and premium offers.
- Loyal Regulars: personalized recommendations and repeat-purchase incentives.
- Budget-Conscious Customers: value bundles and controlled discounts.
- At-Risk Customers: win-back and re-engagement campaigns.

## Run the Project
```bash
pip install -r requirements.txt
python customer_segmentation.py
```

## Files
- `customer_data.csv` — dataset
- `customer_segments.csv` — segmented customer data
- `customer_segmentation.py` — Python implementation
- `Customer_Segmentation_Report.pdf` — submission report
- `Project_Report.html` — browser report
- `outputs/` — charts and segment summary
