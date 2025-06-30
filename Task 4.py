import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
file_path = "C:/Users/heman/OneDrive/Desktop/twitter_validation.csv"
df = pd.read_csv(file_path, delimiter=',', header=None, encoding='unicode_escape')
df.columns = ['Id', 'Entity', 'Sentiment', 'Text']
print("First few rows:")
print(df.head())    
def get_polarity(text):
    return TextBlob(text).sentiment.polarity
df['Polarity'] = df['Text'].apply(get_polarity)
print("\nSummary of polarity:")
print(df['Polarity'].describe())    
plt.hist(df['Polarity'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Sentiment Polarity")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Frequency")
plt.show()