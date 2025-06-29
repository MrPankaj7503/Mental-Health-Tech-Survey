# mental_health_eda

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load the dataset
df = pd.read_csv("mental_health_survey_template.csv")

# ======= BASIC INSPECTION =======
print("Basic Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include="all"))


df = df[(df['Age'] > 10) & (df['Age'] < 100)]

# Optional: normalize gender categories (example)
def clean_gender(g):
    g = str(g).lower()
    if "male" in g:
        return "Male"
    elif "female" in g:
        return "Female"
    else:
        return "Other"
df['Gender'] = df['Gender'].apply(clean_gender)

# ======= VISUALIZATIONS =======

# 1. Treatment Distribution
sns.countplot(data=df, x="treatment")
plt.title("Treatment for Mental Health Condition")
plt.savefig("treatment_distribution.png")
plt.clf()

# 2. Family History vs Treatment
sns.countplot(data=df, x="family_history", hue="treatment")
plt.title("Family History vs Treatment")
plt.savefig("family_history_treatment.png")
plt.clf()

# 3. Age Distribution
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution of Respondents")
plt.savefig("age_distribution.png")
plt.clf()

# 4. Treatment by Gender
sns.countplot(data=df, x="Gender", hue="treatment")
plt.title("Treatment by Gender")
plt.savefig("gender_treatment.png")
plt.clf()

# 5. Country-wise Response Count (Top 10)
df["Country"].value_counts().head(10).plot(kind="barh")
plt.title("Top 10 Countries by Number of Respondents")
plt.savefig("top_countries.png")
plt.clf()

# 6. Work Interference by Treatment
sns.countplot(data=df, x="work_interfere", hue="treatment")
plt.title("Work Interference vs Treatment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("work_interference_treatment.png")
plt.clf()

print("EDA Completed. Charts saved as PNG files in the current directory.")
