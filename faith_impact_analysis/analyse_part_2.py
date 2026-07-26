import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# 1. Load official World Happiness Report 2025
whr = pd.read_excel("../generosity_vs_happiness_model/WHR26_Data_Figure_2.1.xlsx")# Rename country column for merging
whr = whr.rename(columns={'Country name': 'Country',
                          'Life evaluation (3-year average)': 'Life Ladder',
                          'Explained by: Log GDP per capita': 'Log GDP per capita',
                          'Explained by: Social support': 'Social support',
                          'Explained by: Healthy life expectancy': 'Healthy life expectancy at birth',
                          'Explained by: Freedom to make life choices': 'Freedom to make life choices',
                          'Explained by: Generosity': 'Generosity',
                          'Explained by: Perceptions of corruption': 'Perceptions of corruption',
                          })

# Keep only the variables we need
columns_whr = [
    'Country', 'Life Ladder', 'Log GDP per capita',
    'Social support', 'Healthy life expectancy at birth',
    'Freedom to make life choices', 'Generosity',
    'Perceptions of corruption'
]
whr = whr[columns_whr].dropna()

#--------------------------------------------------------------------
# 2. Load extended prayer data and merge on country name
#--------------------------------------------------------------------
prayer = pd.read_csv("../generosity_vs_happiness_model/prayer_data.csv")
data = whr.merge(prayer, on='Country', how='inner')
print(f"Countries in analysis: {len(data)}")

def get_dominant_religion(country):
    if country == "Israel": return "Judaism"
    
    christian_majority = [
        "United States", "United Kingdom", "Germany", "Brazil", "Mexico",
        "Italy", "Spain", "France", "Canada", "Australia", "Argentina",
        "Colombia", "Poland", "Peru", "Chile", "Romania", "Ecuador",
        "Guatemala", "Bolivia", "Dominican Republic", "Haiti", "Honduras",
        "Paraguay", "El Salvador", "Nicaragua", "Costa Rica", "Panama",
        "Uruguay", "Jamaica", "Philippines", "South Africa", "Nigeria",
        "Kenya", "Uganda", "Ghana", "Zambia", "Zimbabwe", "Rwanda",
        "Papua New Guinea", "Austria", "Switzerland", "Ireland", "Portugal", "Greece"
    ]

    if country in christian_majority:
        return "Christianity"
    return "Other_Religion"

data['dominant_religion'] = data['Country'].apply(get_dominant_religion)
data['dominant_religion'] = pd.Categorical(data['dominant_religion'], categories=['Other_Religion', 'Christianity', 'Judaism'])
data_dummies = pd.get_dummies(data, columns=['dominant_religion'], drop_first=True, dtype=int)

interaction_cols = []
for col in data_dummies.columns:
    if col.startswith('dominant_religion_'):
        int_col = f'prayer_x_{col}'
        data_dummies[int_col] = data_dummies['prayer_daily_percent'] * data_dummies[col]
        interaction_cols.append(int_col)

#--------------------------------------------------------------------
# 3. Two-Model Regression: Faith vs. Economics
#--------------------------------------------------------------------

# Group definition
faith_cols = ['prayer_daily_percent'] + [c for c in data_dummies.columns if c.startswith('dominant_religion')] + interaction_cols
economic_cols = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

# Modell A: Pue effect of faith
X_A = sm.add_constant(data_dummies[faith_cols])
model_pure = sm.OLS(data_dummies['Life Ladder'], X_A).fit()

# Modell B: Controlled effect (faith + economics)
X_B = sm.add_constant(data_dummies[faith_cols + economic_cols])
model_full = sm.OLS(data_dummies['Life Ladder'], X_B).fit()

print("\n ===== MODEL A: Pure Effect of Prayer & Religion =====")
print(model_pure.summary())

print("\n===== Model B: Controlled Effect (Adjust for Econ/Social factors) =====")
print(model_full.summary())


# --------------------------------------------------------------------
# 4. Visualization of Interaction Effects
# --------------------------------------------------------------------
print("Creating visualization from model results...")

# The calculated effects from the regression
# Base effecr (Other): -0.0105
# Christianity: -0.0105 + 0.0022 = -0.0083
# Judaism; -0.0105 + 0.0262 = 0.0157
base = model_full.params['prayer_daily_percent']
christian_effect = base + model_full.params.get('prayer_x_dominant_religion_Christianity', 0)
judaism_effect = base + model_full.params.get('prayer_x_dominant_religion_Judaism', 0)

categories = ['Global Average\n(Others)', 'Christian-Majority\nCountries', 'Jewish-Majority(Israel)']
effects = [base, christian_effect, judaism_effect]
colors = ['#e74c3c', '#f39c12', '#2ecc71'] #Red, Orange, Green

plt.figure(figsize=(9, 6))
bars = plt.bar(categories, effects, color=colors)
plt.axhline(0, color='black', linewidth=1.2, linestyle='--')

# Write values directly on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.001 if yval > 0 else -0.002),
             f"{yval:.4f}", ha='center', va ='bottom' if yval > 0 else 'top', fontweight = 'bold', fontsize=11)
    
# Design and labels
plt.title("The Impact of Daily Prayer on Life Satisfaction", fontsize=14, fontweight = 'bold', pad=20)
plt.ylabel('Effect on Life Ladder Score (Regression)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(-0.02, 0.02)
plt.tight_layout()
plt.savefig('faith_impact_visualization.png', dpi=300)
print("Done! Visualization automatically updated and saved.")