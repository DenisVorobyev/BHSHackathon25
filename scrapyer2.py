import requests
from bs4 import BeautifulSoup
import time
import spacy
import json
import cryptoData

# Load spaCy model with word vectors.
nlp = spacy.load("en_core_web_md")

# Debug flag
DEBUG = True

# Helper: set a common header to mimic a browser.
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def is_relevant_semantic(text, target_word, threshold=0.7):
    """
    Tokenizes the text using spaCy and returns True if any token is semantically
    similar to the target_word above the given threshold.
    
    Debug prints similarity scores if DEBUG is True.
    """
    doc = nlp(text)
    target = nlp(target_word)[0]
    for token in doc:
        if token.has_vector:
            sim = token.similarity(target)
            if DEBUG:
                print(f"DEBUG: Token '{token.text}' vs '{target_word}': similarity {sim:.3f}")
            if sim >= threshold:
                return True
    return False

def scrape_twitter(topic, target_word=None, threshold=0.7):
    """
    Scrape Twitter search page for tweets mentioning the topic.
    Optionally filter results during scraping using semantic similarity.
    Note: Twitter may require authentication or API access for a robust solution.
    """
    url = f"https://twitter.com/search?q={topic}&src=typed_query"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print("Error fetching Twitter data:", response.status_code)
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tweets = []
    tweet_divs = soup.find_all("div", attrs={"data-testid": "tweetText"})
    if DEBUG:
        print(f"DEBUG: Found {len(tweet_divs)} tweet elements.")
    
    for tweet_div in tweet_divs:
        text = tweet_div.get_text(separator=" ", strip=True)
        if DEBUG:
            print(f"DEBUG: Raw tweet text: {text[:100]}")  # print first 100 chars
        if target_word is None or is_relevant_semantic(text, target_word, threshold):
            tweets.append(text)
    
    return tweets

def scrape_reddit(topic, target_word=None, threshold=0.7):
    """
    Scrape Reddit search results for posts mentioning the topic using the JSON method.
    Optionally filter results during scraping using semantic similarity.
    """
    url = f"https://www.reddit.com/search.json?q={topic}&restrict_sr=0&sort=new"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print("Error fetching Reddit data:", response.status_code)
        return []
    
    try:
        data = response.json()
        posts = []
        
        # Extract relevant Reddit posts
        for post in data["data"]["children"]:
            post_data = post["data"]
            title = post_data.get("title", "No title")
            selftext = post_data.get("selftext", "")  # Body of the post
            full_text = f"{title} {selftext}".strip()  # Combine title and text
            
            if DEBUG:
                print(f"DEBUG: Raw Reddit post text: {full_text[:100]}")  # First 100 chars
            
            if target_word is None or is_relevant_semantic(full_text, target_word, threshold):
                posts.append(full_text)
        
        return posts
    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
        return []

def scrape_news(topic, target_word=None, threshold=0.7):
    """
    Scrape a news aggregator (Google News in this example) for articles on the topic.
    Optionally filter results during scraping using semantic similarity.
    For robust usage, consider a dedicated API like NewsAPI.
    """
    url = f"https://news.google.com/search?q={topic}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print("Error fetching news data:", response.status_code)
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []
    article_tags = soup.find_all("article")
    if DEBUG:
        print(f"DEBUG: Found {len(article_tags)} news article elements.")
    
    for article in article_tags:
        headline = article.find("h3")
        snippet = article.find("span")
        combined_text = ""
        if headline:
            combined_text += headline.get_text(strip=True) + " "
        if snippet:
            combined_text += snippet.get_text(strip=True)
        combined_text = combined_text.strip()
        if DEBUG:
            print(f"DEBUG: Raw news text: {combined_text[:100]}")
        if combined_text and (target_word is None or is_relevant_semantic(combined_text, target_word, threshold)):
            articles.append(combined_text)
    
    return articles

def scrape(target_word="bitcoin", threshold=0.7):
    """
    Main function to scrape and aggregate data from various sources while filtering 
    results on the fly using semantic similarity.
    """
    print(f"Scraping data related to '{target_word}' and filtering for semantic similarity with '{target_word}'...\n")
    
    twitter_data = scrape_twitter(target_word, target_word, threshold)
    time.sleep(1)  # be polite to servers
    reddit_data = scrape_reddit(target_word, target_word, threshold)
    time.sleep(1)
    news_data = scrape_news(target_word, target_word, threshold)
    
    all_data = twitter_data + reddit_data + news_data
    return all_data

scrapedCryptos = ["bitcoin","ethereum","solana"]
crypto_map = {
    "bitcoin": "btc",
    "ethereum": "eth",  # Note: The correct spelling is "ethereum"
    "solana": "sol"
}

scrapeInterval = 60

if __name__ == "__main__":
    while True:
        for crypto in scrapedCryptos:
            
            results = scrape(crypto)
            data:cryptoData.CryptoInfo = cryptoData.processText(crypto_map[crypto],' '.join(results))
            print(data.SentimentScore, data.Reasoning)
        time.sleep(scrapeInterval)
        
   
