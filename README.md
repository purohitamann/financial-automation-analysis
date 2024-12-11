# Financial Automation & Analysis with LLM

## Overview

**Financial Automation & Analysis with LLM** is a powerful web application that leverages **Large Language Models (LLMs)**, **Pinecone**, and **yfinance** to provide insights and analysis on stocks listed on the **New York Stock Exchange (NYSE)**. This application allows users to:

- Search for stocks using queries like sector, market capitalization, or custom descriptions.
- View detailed stock metrics, including current price, market cap, sector, and industry.
- Analyze price trends with visualizations.
- Filter stocks by sector, market capitalization, and volume.

---

## Features

- **Advanced Search**: Leverages Pinecone and embeddings to deliver stocks related to user queries.
- **Filters**: Easily filter stocks by:
  - Sector
  - Market Capitalization
  - Volume
- **Detailed Stock Insights**:
  - Current Price
  - Market Cap
  - 52-Week High/Low
  - Sector and Industry Information
  - Historical Price Trends
- **Graphical Visualizations**: View 5-day historical stock trends.
- **Backed by AI**: Uses **HuggingFace** embeddings for semantic search and **Groq** for LLM-powered responses.

---

## How It Works

1. **Search Stocks**: Enter your query (e.g., "health care stocks" or "technology sector").
2. **View Stock List**: Get a list of related stocks with summaries.
3. **Filter Results**: Use filters for sector and market capitalization.
4. **Detailed Insights**: Click on a stock to see detailed metrics and historical trends.

---

## Technologies Used

### Backend
- **Python**: Core application logic.
- **YFinance**: For fetching real-time stock data and historical trends.

### AI/ML
- **HuggingFace**: For embedding generation.
- **Pinecone**: Vector database for semantic search.
- **Groq**: For generating intelligent, LLM-based responses.

### Frontend
- **Streamlit**: Interactive user interface for searching, filtering, and visualizing stock data.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip for package installation
- API keys for:
  - **Groq**
  - **Pinecone**

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/purohitamann/financial-automation-analysis.git
   cd financial-automation-analysis
