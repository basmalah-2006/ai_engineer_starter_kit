# Toxic Comment Detector

## Overview

This project is part of **Week 2 – Model Scout Report + Demo** from the AI Engineer Starter Kit.

The objective is to evaluate multiple pre-trained NLP models for toxic comment detection, select the most suitable model based on performance, and build an interactive Gradio application for real-time predictions.

---

## Project Structure

```
week2/
│── app.py
│── inference.py
│── model_scout_comparison.ipynb
│── model_scout_report.md
│── requirements.txt
│── README.md
```

---

## Features

- Compare multiple Hugging Face pre-trained models.
- Evaluate models using accuracy and inference latency.
- Select the best-performing model based on experimental results.
- Interactive Gradio interface for live predictions.
- Simple and lightweight deployment structure.

---

## Candidate Models

- `unitary/toxic-bert`
- `martin-ha/toxic-comment-model`
- `s-nlp/roberta_toxicity_classifier`

---

## Selected Model

**s-nlp/roberta_toxicity_classifier**

### Why was it selected?

The selected model achieved the highest performance during evaluation.

| Metric | Result |
|--------|--------|
| Accuracy | **100%** |
| Average CPU Latency | **40.2 ms / sentence** |

Although the DistilBERT model (`martin-ha/toxic-comment-model`) was faster, the RoBERTa model provided significantly better prediction quality while maintaining reasonable inference speed, making it the best overall choice for a production moderation system.

For detailed evaluation and justification, see:

- `model_scout_report.md`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/basmalah-2006/ai_engineer_starter_kit.git
cd ai_engineer_starter_kit/week2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

A local Gradio interface will open in your browser.

---

## Example Inputs

### Non-toxic

```
Thanks for helping me today!
```

### Toxic

```
You are an idiot and should just quit.
```

---

## Technologies Used

- Python
- Hugging Face Transformers
- Gradio
- PyTorch

---

## Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Gradio application |
| `inference.py` | Loads the selected model and performs inference |
| `model_scout_comparison.ipynb` | Model benchmarking notebook |
| `model_scout_report.md` | Comparison report and model selection rationale |
| `requirements.txt` | Project dependencies |

---

## Notes

The application runs locally using Gradio.

A public Hugging Face Gradio Space could not be created because Hugging Face currently requires a **PRO subscription** to host new Gradio Spaces on free CPU instances. The application has been fully tested and runs correctly in a local environment.

---

## Author

**Basmalah Ahmed**

AI Engineer Starter Kit – Week 2