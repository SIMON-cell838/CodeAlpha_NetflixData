import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
original_url = "https://docs.google.com/spreadsheets/d/1yzr1A6Y6TxjITXzh36zcv4VONsfxjQ8rXCdMIRAcXnE/edit?gid=1154601049#gid=1154601049"

spreadsheet_id = original_url.split('/d/')[1].split('/edit')[0]
gid = original_url.split('gid=')[1].split('#')[0]

csv_export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"

print(f"CSV Export URL: {csv_export_url}")

try:
    df = pd.read_csv(csv_export_url)

    print("\nData successfully read into DataFrame:")
    print(df.head())
    print(f"\nNumber of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

except Exception as e:
    print(f"\nAn error occurred while trying to read the spreadsheet. Please ensure sharing permissions are set correctly (publicly viewable) and the link is correct: {e}")

# Save the DataFrame to a CSV file named 'netflix_content.csv'
df.to_csv("netflix_content.csv", index=False)
print("\nData saved to netflix_content.csv")
#print first 5 rows
print(df.head())
#number of missing values for each column
print(df.isnull().sum())

#For the missing values of Release Date
# Step 1: Find the mode (most frequent Release Date)
mode_release_date = df['Release Date'].mode()[0]  # Gets the first most common date
# Step 2: Fill missing dates with the mode
df['Release Date'] = df['Release Date'].fillna(mode_release_date)
# Step 3: Recalculate 'Release Year' from the updated dates
df['Release Date'] = pd.to_datetime(df['Release Date'], format='%Y-%m-%d')
df['Release Year'] = df['Release Date'].dt.year
# Step 4: Verify no missing values remain
print("Missing values after filling with mode:")
print(df[['Release Date', 'Release Year']].isnull().sum())
#
df['Hours Viewed'] = df['Hours Viewed'].astype(str).str.replace(',','')
df['Hours Viewed'] = pd.to_numeric(df['Hours Viewed'])
df['Content Type'] = df['Content Type'].astype('category')
df['Language Indicator'] = df['Language Indicator'].astype('category')
df['Availbale Globally?'] = df['Available Globally?'].astype('category')
df['Title'] = df['Title'].astype('category')

#Trends 
#number of Titles added per year
Release_Year_Counts = df['Release Year'].value_counts().sort_index()
plt.figure(figsize=(12,12))
sns.lineplot(x=Release_Year_Counts.index, y=Release_Year_Counts.values)
plt.title('Number of Netflix Title Released Per Year')
plt.xlabel('Release Year')
plt.ylabel('Number Of Titles')
plt.grid(True)
plt.show()

#Patterns
#Distribution of Content Types on Netflix

plt.figure(figsize=(8,8))
df['Content Type'].value_counts().plot.pie(autopct='%1.1f%%',startangle=90,colors=['skyblue','lightcoral'])
plt.title("Distribution of Content Types on Netflix")
plt.ylabel('')
plt.show()

#Anomalies
#Hours Viewed

top_viewed_titles = df.sort_values(by='Hours Viewed', ascending=False).head(10)
print("\nTop 10 Titles by Hours Viewed (Potential Anomalies/Hits):")
print(top_viewed_titles[['Title', 'Content Type', 'Release Date', 'Hours Viewed']])
current_year = pd.Timestamp.now().year
recent_low_performers = df[(df['Release Year'] <= current_year - 5) & (df['Hours Viewed'] < 300000)] #***
print("\nRecent Titles with Low Hours Viewed (Potential Underperformers/Anomalies):")
print(recent_low_performers[['Title', 'Content Type', 'Release Date', 'Hours Viewed']].head())
total_low_hours_titles = len(recent_low_performers)
#total_low_hours_titles = recent_low_performers.shape[0]
print("Total Number of low Hours Viewed Titles: ",total_low_hours_titles)

#patterns
#Distribution of Content by Language Indicator
plt.figure(figsize=(10, 7))
sns.countplot(data=df, y='Language Indicator', order=df['Language Indicator'].value_counts().index,palette='viridis')
plt.title('Overall Distribution of Content by Language Indicator')
plt.xlabel('Number of Titles')
plt.ylabel('Language Indicator')
plt.tight_layout()
plt.show()
# To see the exact counts:
print("\nTop 10 Languages by Number of Titles:")
print(df['Language Indicator'].value_counts().head(10))

#Hypothesis

print("\n--- Testing Hypothesis 1: TV Shows generate more hours viewed than Movies ---")

# Calculate total hours viewed per Content Type
total_hours_by_type = df.groupby('Content Type')['Hours Viewed'].sum()
print("\nTotal Hours Viewed by Content Type:")
print(total_hours_by_type)

plt.figure(figsize=(8, 6))
sns.boxplot(x='Content Type', y='Hours Viewed', data=df)
plt.title('Distribution of Hours Viewed by Content Type')
plt.xlabel('Content Type')
plt.ylabel('Hours Viewed')
plt.yscale('log') # Use log scale if hours viewed vary wildly
plt.show()

# Perform Mann-Whitney U test (more robust for non-normal, skewed data like viewing hours)
movies_hours = df[df['Content Type'] == 'Movie']['Hours Viewed']
tv_shows_hours = df[df['Content Type'] == 'TV Show']['Hours Viewed']

if not movies_hours.empty and not tv_shows_hours.empty:
    stat, p = stats.mannwhitneyu(tv_shows_hours, movies_hours, alternative='greater') # Testing if TV shows are 'greater'
    print(f"\nMann-Whitney U Test:")
    print(f"Statistic = {stat:.2f}, P-value = {p:.3e}") # Print p-value in scientific notation

    alpha = 0.05 # Significance level
    if p < alpha:
        print(f"Validation: Reject null hypothesis. TV Shows statistically generate more hours viewed than Movies (p < {alpha}).")
    else:
        print(f"Validation: Fail to reject null hypothesis. No statistically significant evidence that TV Shows generate more hours viewed than Movies (p >= {alpha}).")
else:
    print("Validation: Not enough data for one or both content types to perform statistical test.")

print("\n--- Testing Hypothesis 2: Non-English content growing in popularity ---")

# Define English vs. Non-English (assuming 'English' is the indicator for English)
df['Is English'] = df['Language Indicator'].apply(lambda x: 'English' if x == 'English' else 'Non-English')

# Group by Release Year and Is English, sum Hours Viewed
language_popularity_yearly = df.groupby(['Release Year', 'Is English'])['Hours Viewed'].sum().unstack(fill_value=0)

# Normalize to show proportion if total content size changes drastically
language_popularity_yearly_norm = language_popularity_yearly.div(language_popularity_yearly.sum(axis=1), axis=0)

# Plot absolute hours viewed trend
plt.figure(figsize=(12, 7))
sns.lineplot(data=language_popularity_yearly, markers=True)
plt.title('Total Hours Viewed by English vs. Non-English Content Per Year')
plt.xlabel('Release Year')
plt.ylabel('Total Hours Viewed')
plt.legend(title='Language Category')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot proportional trend
plt.figure(figsize=(12, 7))
sns.lineplot(data=language_popularity_yearly_norm, markers=True)
plt.title('Proportion of Total Hours Viewed by English vs. Non-English Content Per Year')
plt.xlabel('Release Year')
plt.ylabel('Proportion of Total Hours Viewed')
plt.legend(title='Language Category')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Validate assumption: Check recent trend
# Get the trend for Non-English proportion
non_english_trend = language_popularity_yearly_norm['Non-English'].diff().dropna()
if not non_english_trend.empty:
    average_non_english_growth = non_english_trend.mean()
    print(f"Average yearly change in Non-English content's proportion of hours viewed: {average_non_english_growth:.4f}")

    if average_non_english_growth > 0:
        print("Validation: The data suggests a positive trend for Non-English content's popularity.")
    else:
        print("Validation: The data does NOT strongly suggest a positive trend for Non-English content's popularity.")
else:
    print("Validation: Not enough data points to determine a trend for Non-English content proportion.")

df.drop(columns=['Is English'], inplace=True) # Clean up temporary column

# Plot : Hours Viewed by Global Availability
global_availability_col = None
if 'Availbale Globally?' in df.columns:
        global_availability_col = 'Availbale Globally?'
elif 'Available Globally?' in df.columns:
        global_availability_col = 'Available Globally?'

if global_availability_col and 'Hours Viewed' in df.columns and pd.api.types.is_numeric_dtype(df['Hours Viewed']) and not df[[global_availability_col, 'Hours Viewed']].empty:
        global_hours = df.groupby(global_availability_col)['Hours Viewed'].sum().reset_index()

        if not global_hours.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(x=global_availability_col, y='Hours Viewed', data=global_hours, palette='rocket')
            plt.title('Total Hours Viewed by Global Availability')
            plt.xlabel('Available Globally?')
            plt.ylabel('Total Hours Viewed (Log Scale)')
            plt.yscale('log')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            print("Plot 'hours_viewed_by_global_availability'")
        else:
            print("No data available to generate 'Hours Viewed by Global Availability' plot.")
else:
        print("Required columns for Plot are not available, not numeric, or empty.")
        if not global_availability_col:
            print(f"Check if '{'Availbale Globally?' if 'Availbale Globally?' not in df.columns else 'Available Globally?'}' column exists for Plot 8.")

# Plot : Average Hours Viewed per Title per Year by Content Type
if 'Release Year' in df.columns and 'Content Type' in df.columns and 'Hours Viewed' in df.columns and \
    pd.api.types.is_numeric_dtype(df['Hours Viewed']) and not df[['Release Year', 'Content Type', 'Hours Viewed']].empty:

        avg_hours_per_year_type = df.groupby(['Release Year', 'Content Type'])['Hours Viewed'].mean().unstack(fill_value=0)

        if not avg_hours_per_year_type.empty:
            plt.figure(figsize=(12, 7))
            sns.lineplot(data=avg_hours_per_year_type, markers=True, dashes=False)
            plt.title('Average Hours Viewed Per Title Per Year by Content Type')
            plt.xlabel('Release Year')
            plt.ylabel('Average Hours Viewed (Log Scale)')
            plt.yscale('log')
            plt.legend(title='Content Type')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            print("Plot 'avg_hours_per_title_per_year'")
        else:
            print("No data available to generate 'Average Hours Viewed per Title per Year' plot.")
else:
        print("Required columns for Plot are not available, not numeric, or empty.")

# Calculate percentiles to define sentiment thresholds
low_threshold = df['Hours Viewed'].quantile(0.33)  # Bottom 33%
high_threshold = df['Hours Viewed'].quantile(0.66)  # Top 33%

def assign_sentiment(viewership):
    if viewership >= high_threshold:
        return "Positive"
    elif viewership <= low_threshold:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['Hours Viewed'].apply(assign_sentiment)
sentiment_by_type = df.groupby(['Content Type', 'Sentiment']).size().unstack()
print(sentiment_by_type)
# Sentiment distribution by Content Type
sns.countplot(data=df, x='Content Type', hue='Sentiment', palette=['dodgerblue', 'gray', 'darkorange'])
plt.title("Sentiment by Content Type (Movie vs. Show)")
plt.show()

# Categorize into tiers (Low/Medium/High)
df['Engagement_Tier'] = pd.qcut(
    df['Hours Viewed'],
    q=[0, 0.25, 0.75, 1],
    labels=['Low', 'Medium', 'High']
)
emotion_rules = {
    ('English', 'High'): 'Joy',
    ('English', 'Low'): 'Sadness',
    ('Korean', 'High'): 'Excitement',
    ('Korean', 'Low'): 'Disappointment',
    ('Non-English', 'High'): 'Interest',
    ('Non-English', 'Low'): 'Indifference'
}

df['Behavioral_Emotion'] = df.apply(
    lambda row: emotion_rules.get((row['Language Indicator'], row['Engagement_Tier']), 'Neutral'),
    axis=1
)
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x='Language Indicator',
    hue='Behavioral_Emotion',
    palette='viridis',
    edgecolor='black'
)
plt.title('Emotional Engagement by Language Indicator')
plt.xlabel('Language')
plt.ylabel('Number of Titles')
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
