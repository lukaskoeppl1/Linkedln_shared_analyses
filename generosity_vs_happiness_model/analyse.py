import pandas as pd
import matplotlib.pyplot as plt
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

#--------------------------------------------------------------------
# 3. Multiple regression: Base model + extended model
#--------------------------------------------------------------------

# Model 1: WHR control factors (including Generosity, NO prayer)
X_base = data[['Log GDP per capita', 'Social support',
           'Healthy life expectancy at birth', 'Freedom to make life choices',
           'Generosity', 'Perceptions of corruption']]
X_base_const = sm.add_constant(X_base)
model_base = sm.OLS(data['Life Ladder'], X_base_const).fit()
print("\n" + "=" * 55)
print("\n===== Model 1: Base WHR Factors (%) =====")
print(model_base.summary())

# Model 2: WHR control factors + daily prayer
X_extended = data[['Log GDP per capita', 'Social support',
           'Healthy life expectancy at birth', 'Freedom to make life choices',
           'Generosity', 'Perceptions of corruption', 'prayer_daily_percent']]
X_extended_const = sm.add_constant(X_extended)
model_extended = sm.OLS(data['Life Ladder'], X_extended_const).fit()
print("\n" + "=" * 55)
print("\n===== Model 2: Granular prayer variables =====")
print(model_extended.summary())

#--------------------------------------------------------------------
# 4. Coefficient comparison plot (Model 2)
#--------------------------------------------------------------------
# We drop 'prayer_daily_percent' from the plot to focus on the other variables
# on Generosity vs. Economy, as requested by the user.
coefs = model_extended.params.drop(['const', 'prayer_daily_percent'])
errors = model_extended.bse.drop(['const', 'prayer_daily_percent'])

plt.figure(figsize=(10,6))
coefs.sort_values().plot(kind='barh', xerr=errors, capsize=5,
                         color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.title('The True Driver of Happiness: Generosity vs. Economy')
plt.xlabel("Influence on Happiness Score")
plt.tight_layout()
plt.savefig('coefficients_model2.png', dpi=300, bbox_inches = 'tight', pad_inches=0.3)
print("\nPlot saved as 'coefficients_model2.png'")
plt.show()