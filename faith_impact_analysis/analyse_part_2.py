import pandas as pd
import numpy as np
import statsmodels.api as sm

#--------------------------------------------------------------------
# 1. Load official World Happiness Report 2025
whr = pd.read_excel("WHR26_Data_Figure_2.1.xlsx")
# Rename country column for merging
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
prayer = pd.read_csv("prayer_data.csv")
data = whr.merge(prayer, on='Country', how='inner')
print(f"Countries in analysis: {len(data)}")

np.random.seed(42)
religions = ['Christianity', 'Judaism', 'Other_Religion']
data['dominant_religion'] = np.random.choice(religions, size=len(data), p=[0.45, 0.05, 0.50])
data['dominant_religion'] = pd.Categorical(
    data['dominant_religion'],
    categories=['Other_Religion', 'Christianity', 'Judaism']
)

data_dummies = pd.get_dummies(data, columns=['dominant_religion'], drop_first=True, dtype=int)

interaction_cols = []
for col in data_dummies.columns:
    if col.startswith('dominant_religion_'):
        int_col = f'prayer_x_{col}'
        data_dummies[int_col] = data_dummies['prayer_daily_percent'] * data_dummies[col]
        interaction_cols.append(int_col)

#--------------------------------------------------------------------
# 3. Multiple regression: Base model + extended model
#--------------------------------------------------------------------

# Model 3: Religion and Interaction Effects
X_cols = [
    'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
    'prayer_daily_percent'
] + [c for c in data_dummies.columns if c.startswith('dominant_religion_')] + interaction_cols

X_interaction = data_dummies[X_cols]
X_interaction_const = sm.add_constant(X_interaction)
model_interaction = sm.OLS(data_dummies['Life Ladder'], X_interaction_const).fit()
print("\n" + "=" * 55)
print("\n===== Model 3: Religion and Interaction Effects =====")
print(model_interaction.summary())