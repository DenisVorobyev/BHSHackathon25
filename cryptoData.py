import pickle
import google.generativeai as genai
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

analyzer = SentimentIntensityAnalyzer()
class CryptoInfo:
    def __init__(self, SentimentScore, Reasoning):
        self.SentimentScore = SentimentScore
        self.Reasoning = Reasoning

def processText(name: str, text: str):
    """
    Processes the given text by summarizing it (translating if needed),
    analyzing sentiment, and providing reasoning.
    Stores the result in a pickle file.
    """
    
    # Gemini - Summarization & Translation
    summary_prompt = (
        "You are a helpful assistant. Read the provided text and translate any non-English text to English. "
        "Also, remove blatant sarcasm while retaining the original meaning of the text."
    )
    summary_response = genai.GenerativeModel("gemini-2.0-flash").generate_content([summary_prompt, text])
    summary = summary_response.text if summary_response else text  # Fallback to original text if Gemini fails
    
    # Sentiment Analysis
    SentimentScore = analyzer.polarity_scores(summary)
    

    # Gemini - Reasoning for Sentiment Score
    reasoning_prompt = (
        f"You are given a sentiment score followed by a text that received this score. "
        f"Higher values are more positive, while lower values are more negative. "
        f"Given the sentiment score Positive:{SentimentScore['pos']}, Negative:{SentimentScore['neg']}, Neutral:{SentimentScore['neu']} justify it with reasoning from the text in a short but detail heavy paragraph. Do not include the sentiment score in your response."
    )
    reasoning_response = genai.GenerativeModel("gemini-2.0-flash").generate_content([reasoning_prompt, text])
    reasoning = reasoning_response.text if reasoning_response else "No reasoning available."

    # Store data
    info = CryptoInfo(SentimentScore, reasoning)
    with open(f"Datastore/{name}.pkl", "wb") as f:
        pickle.dump(info, f)

    return info

def getCryptoInfo(name: str):
    """
    Retrieves stored CryptoInfo from a pickle file.
    """
    with open(f"Datastore/{name}.pkl", "rb") as f:  # Fixed mode: "rb" instead of "wb"
        info = pickle.load(f)
        return info
