# CodeAlpha_NetflixData
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
