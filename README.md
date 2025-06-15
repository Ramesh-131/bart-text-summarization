# bart-text-summarization
A comparative analysis of extractive and abstractive summarization techniques using traditional methods and BART transformer model.

# BART Text Summarization

## Overview
This project presents a comparative analysis between extractive and abstractive text summarization techniques using traditional algorithms like TF-IDF and TextRank and transformer-based models like BART.

## Techniques Used
- TF-IDF, TextRank, LexRank (Extractive)
- BART Fine-Tuning using Hugging Face Transformers (Abstractive)

## Dataset
- CNN/DailyMail dataset (for BART training)
- Custom text inputs for extractive methods

## Results
- ROUGE and BLEU scores show improved performance using BART
- BART-generated summaries were more fluent and context-aware

## Project Files
- `bart_summarization.ipynb`: Main code notebook
- `report/BART_Summarization_IEEE_Format.pdf`: Final formatted research paper
- `requirements.txt`: Python libraries required

## How to Run
```bash
pip install -r requirements.txt
