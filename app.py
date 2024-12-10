# align's the message to the right
import streamlit as st
import streamlit_shadcn_ui as ui
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os
load_dotenv()


import os
# !pip install groq
from groq import Groq
client = Groq(
  api_key=os.getenv('GROQ_API_KEY'),
)
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
pinecone_index = pc.Index('stocks')

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)


def cosine_similarity_between_sentences(sentence1, sentence2):
    """
    Calculates the cosine similarity between two sentences.

    Args:
        sentence1 (str): The first sentence for similarity comparison.
        sentence2 (str): The second sentence for similarity comparison.

    Returns:
        float: The cosine similarity score between the two sentences,
               ranging from -1 (completely opposite) to 1 (identical).

    Notes:
        Prints the similarity score to the console in a formatted string.
    """
    # Get embeddings for both sentences
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))

    # Reshape embeddings for cosine_similarity function
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    similarity_score = similarity[0][0]
    print(f"Cosine similarity between the two sentences: {similarity_score:.4f}")
    return similarity_score


# # Example usage
# sentence1 = "I like walking to the park"
# sentence2 = "I like running to the office"

# similarity = cosine_similarity_between_sentences(sentence1, sentence2)

def prepare_augmented_query(query):
  raw_query_embedding = get_huggingface_embeddings(query)
  top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace='stock-descriptions')
  contexts = [item['metadata']['text'] for item in top_matches['matches']]
  augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
  return augmented_query

def perform_rag(query):
  augmented_query = prepare_augmented_query(query)
  system_prompt = f"""You are an expert at providing answers about stocks. Please answer my question provided.
  """

  llm_response = client.chat.completions.create(
      model="llama-3.1-70b-versatile",
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": augmented_query}
      ]
  )

  response = llm_response.choices[0].message.content

  return response

col1, col2 = st.columns([1,2])


with col1:
  st.title("Find me a STOCK!")
  st.write('Enter the name of the stock you want to know more about')
  input_value = st.text_input('What Stocks are you interested in?', placeholder="give me tech stocks", key="input1")

with col2:
  st.write("you entered:", input_value)
  
  result = perform_rag(prepare_augmented_query(input_value))

  st.write("result:", result)
# Input Component
