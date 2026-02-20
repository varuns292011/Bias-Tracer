# Bias Amplification Tracer
### Quantifying How AI Personas Amplify Bias in Sequential Text Processing

## Purpose
AI models are increasingly used to summarize, rewrite, and redistribute content across news platforms, social media, and newsletters. But what happens to the original meaning when text passes through multiple AI processing stages? This project answers that question with measurable, reproducible data.

Bias Amplification Tracer takes any neutral text input and passes it through four distinct AI personas sequentially, measuring how sentiment and ideological bias change at each step using NLP sentiment analysis.

## What It Does
1. User inputs any text — news article, historical fact, statement
2. Text passes through four AI personas sequentially:
   - **Neutral** — objective journalist
   - **Left** — progressive commentator
   - **Right** — conservative commentator
   - **Tabloid** — sensationalist writer
3. Sentiment score measured at each step using TextBlob
4. Results visualized as a bias drift chart

## Key Finding
Even a completely neutral historical fact produces wildly different sentiment scores across personas. The neutral AI alone introduced a 0.40 positive drift from a 0.00 baseline — proving that prompt style alone significantly amplifies bias before any political framing is applied.

## Sample Output
![Bias Amplification Chart](bias_amplification.png)

## Conclusions
- Neutral AI prompting alone introduces measurable positive sentiment drift
- Political personas swing sentiment in predictable but significant ways
- Tabloid rewriting consistently produces the most negative sentiment
- Sequential AI processing of the same content produces compounding bias amplification
- Prompt style is a primary driver of bias in LLM outputs, independent of factual content

## How To Run
Install dependencies:
```bash
pip install groq textblob matplotlib pandas
python -m textblob.download_corpora
```
Run:
```bash
python3 bias-tracer.py
```
Enter any text when prompted.

## Tech Stack
Python, Groq LLM, TextBlob, Matplotlib, Pandas, Git