import os
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from textblob import TextBlob

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def process_through_persona(text, persona):
    prompts = {
        "neutral": "You are a neutral journalist. Summarize this text factually with no opinion:",
        "left": "You are a progressive political commentator. Summarize this text from your perspective:",
        "right": "You are a conservative political commentator. Summarize this text from your perspective:",
        "tabloid": "You are a sensationalist tabloid writer. Rewrite this text dramatically:"
    }
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"{prompts[persona]}\n\n{text}"}],
        max_tokens=300
    )
    return response.choices[0].message.content

original_text = input("Enter your text to analyze: ").strip()

print("🔍 BIAS AMPLIFICATION TRACER")
print("="*50)
print(f"ORIGINAL TEXT sentiment: {get_sentiment(original_text):.4f}")
print()

personas = ["neutral", "left", "right", "tabloid"]
results = []
current_text = original_text

for persona in personas:
    print(f"Processing through {persona.upper()} persona...")
    output = process_through_persona(current_text, persona)
    sentiment = get_sentiment(output)
    results.append({
        "persona": persona,
        "sentiment": sentiment,
        "text": output
    })
    print(f"Sentiment score: {sentiment:.4f}")
    print(f"Output: {output[:150]}...")
    print()
    current_text = output

# Visualize
df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(df['persona'].tolist(), df['sentiment'].tolist(), marker='o', linewidth=2, color='red', markersize=8)
plt.axhline(y=get_sentiment(original_text), color='blue', linestyle='--', label='Original sentiment')
plt.title('Bias Amplification Across AI Personas', fontsize=14, fontweight='bold')
plt.xlabel('Persona')
plt.ylabel('Sentiment Score (-1 negative to +1 positive)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bias_amplification.png', dpi=150)
print("Chart saved as bias_amplification.png")
plt.show()