# Daily Quote Sentiment Analyzer

## Overview

This project is part of **Week 1 — Engineering Foundations for AI** from the **AI Engineer Starter Kit**.

The notebook demonstrates the complete workflow of calling an external REST API, retrieving text data, and analyzing its sentiment using a Hugging Face Transformer model.

## Features

- Fetches a random quote from a public REST API
- Performs sentiment analysis using Hugging Face Transformers
- Demonstrates JSON parsing
- Uses Python `requests`
- Runs entirely inside a Jupyter Notebook
- Reproducible environment using `requirements.txt`

## Technologies

- Python
- Requests
- Hugging Face Transformers
- Jupyter Notebook

## Project Files

| File | Description |
| --- | --- |
| `daily_quote_sentiment.ipynb` | Main notebook |
| `requirements.txt` | Project dependencies |
| `README.md` | Project documentation |

## Installation

Clone the repository:

```bash
git clone https://github.com/basmalah-2006/ai_engineer_starter_kit.git
cd ai_engineer_starter_kit/week1
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the notebook:

```bash
jupyter notebook daily_quote_sentiment.ipynb
```

Then run all notebook cells from top to bottom.

## Learning Outcomes

This project demonstrates:

- Virtual environments
- REST API integration
- JSON handling
- Hugging Face `pipeline()`
- Jupyter Notebook workflow
- Clean project organization

## Author

**Basmalah Ahmed**  
AI Engineer Starter Kit — Week 1