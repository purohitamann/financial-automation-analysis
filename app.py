import streamlit as st
import json
import yfinance as yf
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import requests
from pinecone import Pinecone
import os

# Load environment variables
load_dotenv()

# Initialize Groq and Pinecone
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("stocks")


# Function to get HuggingFace embeddings
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generate embeddings for the given text using Hugging Face models.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)


# Function to calculate cosine similarity
def cosine_similarity_between_sentences(sentence1, sentence2):
    """
    Compute the cosine similarity between two sentences.
    """
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


# Function to prepare augmented query
def prepare_augmented_query(query):
    """
    Create an augmented query by finding similar context from Pinecone.
    """
    raw_query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=15,
        include_metadata=True,
        namespace="stock-descriptions",
    )
    contexts = [item["metadata"]["text"] for item in top_matches["matches"]]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\nMY QUESTION:\n" + query
    return augmented_query


# Function to perform RAG
def perform_rag(query):
    """
    Perform Retrieval-Augmented Generation (RAG) to retrieve relevant stocks.
    """
    augmented_query = prepare_augmented_query(query)
    system_prompt = """
    You are an expert at providing answers about stocks. Return only a valid JSON object.
    Do not include explanations or preamble. Follow this format:
    [
        {"Name": "eHealth, Inc.", "Ticker": "EHTH", "Description": "Provides private online health insurance services."},
        {"Name": "UnitedHealth Group", "Ticker": "UNH", "Description": "Operates as a diversified healthcare company."},
        {"Name": "Elevance Health", "Ticker": "ELV", "Description": "Operates as a healthcare benefits company."},
        ....
    ]
    """

    llm_response = client.chat.completions.create(
        model="llama-3.3-70b-specdec",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query},
        ],
    )

    response = llm_response.choices[0].message.content

    try:
        parsed_response = json.loads(response)  # Ensure it is valid JSON
        validate_response(parsed_response)
        return parsed_response
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error parsing LLM response: {e}")
        raise ValueError("The LLM response is not valid JSON.")


# Function to validate the response schema
def validate_response(response):
    """
    Validate the schema of the response from the LLM.
    """
    if not isinstance(response, list):
        raise ValueError("Response is not a list.")
    required_keys = {"Name", "Ticker", "Description"}
    for stock in response:
        if not isinstance(stock, dict) or not required_keys.issubset(stock.keys()):
            raise ValueError("Invalid stock data structure.")
    return True


# Function to fetch stock data from yfinance
@st.cache_data
def get_stock_data(ticker):
    """
    Fetch detailed stock information using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Name": info.get("shortName", "N/A"),
            "Symbol": ticker.upper(),
            "Current Price": stock.history(period="1d").iloc[-1]["Close"],
            "Market Cap": info.get("marketCap", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Website": info.get("website", "N/A"),
            "Prices": stock.history(period="5d"),
            "Description": info.get("longBusinessSummary", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
        }
    except Exception as e:
        return {"Error": str(e)}


# Function to display stock details
def display_stock_details(ticker):
    """
    Display detailed information about a specific stock.
    """
    
 
    stock = get_stock_data(ticker)
    st.header(f"{ticker['Name']} ({ticker['Ticker']})")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Symbol:** {ticker['Symbol']}") 
        st.write(f"**Current Price:** ${ticker.get('Current Price', 'N/A')}")
        st.write(f"**Market Cap:** {ticker.get('Market Cap', 'N/A')}")
        st.write(f"**Industry:** {ticker.get('Industry', 'N/A')}")
      
    with col2:
        st.write(f"**P/E Ratio:** {ticker.get('P/E Ratio', 'N/A')}")
        st.write(f"**Sector:** {ticker.get('Sector', 'N/A')}")
        st.write(f"**52 Week High:** ${ticker.get('52 Week High', 'N/A')}")
        st.write(f"**52 Week Low:** ${ticker.get('52 Week Low', 'N/A')}")

    st.write(f"**Description:** {ticker['Description']}")
    
   
  
    
    st.subheader("Price Trends (Last 5 Days)")
    prices = stock.get("Prices")
    st.write(ticker.get("Prices"))

    st.write(f"[Visit Website]({stock.get('Website', '#')})")

    if st.button("Back to List"):
        del st.session_state.selected_stock
       

# Function to add filters
def add_filters():
    """
    Add sidebar filters for sectors and market capitalization.
    """
    st.sidebar.header("Filter Stocks")
    sectors = ["All", "Technology", "Health Care", "Finance", "Energy"]
    selected_sector = st.sidebar.selectbox("Sector", sectors)

    min_market_cap = st.sidebar.slider("Minimum Market Cap (in billions)", 0, 1000, 10)
    max_market_cap = st.sidebar.slider("Maximum Market Cap (in billions)", 10, 1000, 500)

    return selected_sector, min_market_cap * 1e9, max_market_cap * 1e9


# Function to display the stock list
def display_stock_list():
    """
    Display a list of stocks with filters and options to view details.
    """
    selected_sector, min_market_cap, max_market_cap = add_filters()
    st.header("Find me a STOCK!")
    input_value = st.text_input("What Stocks are you interested in?", placeholder="e.g., health care", key="input1")

    if input_value:
        try:
            stockslist = perform_rag(input_value)
            filtered_stocks = [
                stock for stock in stockslist
                if (selected_sector == "All" or stock.get("Sector") == selected_sector)
                or min_market_cap <= stock.get("Market Cap", 0) <= max_market_cap
            ]

            cols = st.columns(3)
            if len(filtered_stocks) == 0:
                st.write("No results found.")
                return
            for idx, stock in enumerate(filtered_stocks):
                with cols[idx % 3]:
                    st.markdown(f"**{stock['Name']} ({stock['Ticker']})**")
                    if st.button(f"View {stock['Name']} ({stock['Ticker']})", key=f"button_{idx}"):
                        st.session_state.selected_stock = stock
                      
        except ValueError as e:
            st.error(f"Error: {e}")


# App header
def header():
    """
    Display the app header.
    """
    st.markdown(
        """
        <div style="
            background: rgba(0, 123, 255, 0.1); 
            border: 1px solid rgba(0, 123, 255, 0.5); 
            border-radius: 8px; 
            padding: 16px; 
            margin-bottom: 16px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
            backdrop-filter: blur(5px);
        ">
            <h1 style="margin: 0; color: #007BFF;">Financial Analysis & Automation with LLM</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Main logic
header()
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

if st.session_state.selected_stock:
    stock = st.session_state.selected_stock
    detailed_stock_data = get_stock_data(stock["Ticker"])
    stock.update(detailed_stock_data)
    display_stock_details(stock)
else:
    display_stock_list()
