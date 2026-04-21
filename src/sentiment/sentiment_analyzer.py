import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

class NewsSentimentAnalyzer:
    """
    Fetches news headlines for a stock ticker and calculates sentiment using VADER.
    Includes a fallback/simulation method to generate proxy sentiments for academic demo purposes 
    if live scraping fails.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def fetch_recent_headlines(self, ticker, limit=10):
        """
        Attempts to scrape recent news headlines for a ticker.
        Note: Simple scraping is heavily rate-limited or blocked by major financial sites.
        This provides a basic Google News RSS approach.
        """
        headlines = []
        try:
            # Basic Google News RSS feed search
            url = f"https://news.google.com/rss/search?q={ticker}+stock"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "xml")
                items = soup.find_all("item", limit=limit)
                for item in items:
                    headlines.append(item.title.text)
        except Exception as e:
            print(f"Error fetching live headlines: {e}")
            
        return headlines

    def analyze_headlines(self, headlines):
        """
        Given a list of headlines, returns a single aggregate compound sentiment score.
        Score ranges from -1 (Extremely Negative) to 1 (Extremely Positive).
        """
        if not headlines:
            return 0.0
            
        scores = []
        for hl in headlines:
            scores.append(self.analyzer.polarity_scores(hl)['compound'])
            
        return np.mean(scores)
        
    def get_simulated_historical_sentiment(self, df):
        """
        Generates simulated daily sentiment scores based on price momentum.
        Use this to train the ensemble model historically, as fetching 5 years of historical
        daily news headlines is not feasible without expensive commercial APIs.
        
        Args:
            df (pd.DataFrame): Dataframe with at least 'Returns' calculated.
        """
        if 'Returns' not in df.columns:
            df = df.copy()
            df['Returns'] = df['Close'].pct_change()
            
        # Add some noise to the returns to act as a simulated sentiment signal
        np.random.seed(42)
        noise = np.random.normal(0, 0.2, len(df))
        
        # Scale returns to somewhat map to a [-1, 1] range, added with noise
        # This creates a feature correlated conceptually with momentum (good news = price up)
        sentiment = df['Returns'].fillna(0) * 10 
        sentiment = sentiment + noise
        
        # Clip to VADER's compound range [-1, 1]
        df['Sentiment_Score'] = np.clip(sentiment, -1, 1)
        return df

# Example Block
if __name__ == "__main__":
    analyzer = NewsSentimentAnalyzer()
    
    ticker = "AAPL"
    print(f"Fetching live sentiment for {ticker}...")
    hl = analyzer.fetch_recent_headlines(ticker, limit=5)
    for h in hl:
        print(f"- {h}")
    score = analyzer.analyze_headlines(hl)
    print(f"\nLive Aggregate Compound Sentiment Score: {score:.4f}")
