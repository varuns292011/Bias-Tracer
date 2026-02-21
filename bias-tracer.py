import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from textblob import TextBlob
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

TOPICS = [
    "The government raised taxes on middle class families this year.",
    "Scientists confirmed climate change is accelerating faster than predicted.",
    "The unemployment rate dropped to its lowest level in a decade.",
    "A new immigration policy was signed into law yesterday.",
    "The supreme court ruled on a major gun control case today.",
    "Tech companies reported record profits while wages stagnated.",
    "A new study shows social media increases depression in teenagers.",
    "The military budget was increased by 15% this fiscal year.",
    "Free college tuition was proposed in the new education bill.",
    "Police departments received additional funding for new equipment.",
    "Renewable energy now powers 30% of the national grid.",
    "The minimum wage increase bill was passed by congress.",
    "A new healthcare bill would expand coverage to millions.",
    "Border security spending doubled under the new administration.",
    "Public school funding was cut in the latest budget proposal.",
    "A billionaire paid zero taxes due to legal loopholes.",
    "Abortion restrictions were expanded in three new states.",
    "The stock market reached an all time high this quarter.",
    "Foreign aid was reduced by 20% in the new budget.",
    "A new drug was approved that could cure a common disease."
]

PERSONAS = {
    "neutral": "You are a neutral journalist. Summarize this text factually with no opinion in 2-3 sentences:",
    "left": "You are a progressive political commentator. Summarize this text from your perspective in 2-3 sentences:",
    "right": "You are a conservative political commentator. Summarize this text from your perspective in 2-3 sentences:",
    "tabloid": "You are a sensationalist tabloid writer. Rewrite this text dramatically in 2-3 sentences:"
}

def get_sentiment(text):
    analysis = TextBlob(text)
    return round(analysis.sentiment.polarity, 4)

def get_subjectivity(text):
    analysis = TextBlob(text)
    return round(analysis.sentiment.subjectivity, 4)

def process_through_persona(text, persona):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"{PERSONAS[persona]}\n\n{text}"}],
        max_tokens=200
    )
    return response.choices[0].message.content

print("🔍 BIAS AMPLIFICATION TRACER — FULL STUDY")
print("="*60)
print(f"Testing {len(TOPICS)} topics across {len(PERSONAS)} personas...")
print()

results = []

for i, topic in enumerate(TOPICS):
    print(f"Processing topic {i+1}/{len(TOPICS)}: {topic[:50]}...")
    original_sentiment = get_sentiment(topic)
    
    for persona in PERSONAS:
        output = process_through_persona(topic, persona)
        sentiment = get_sentiment(output)
        subjectivity = get_subjectivity(output)
        drift = round(sentiment - original_sentiment, 4)
        
        results.append({
            "topic": topic[:40] + "...",
            "persona": persona,
            "original_sentiment": original_sentiment,
            "output_sentiment": sentiment,
            "subjectivity": subjectivity,
            "drift": drift,
            "output": output
        })

df = pd.DataFrame(results)
df.to_csv("bias_results.csv", index=False)
print("\n✅ Results saved to bias_results.csv")

# Statistical significance test
print("\nSTATISTICAL SIGNIFICANCE RESULTS")
print("="*60)
for persona in PERSONAS:
    persona_data = df[df['persona'] == persona]['drift']
    t_stat, p_value = stats.ttest_1samp(persona_data, 0)
    significance = "SIGNIFICANT ✅" if p_value < 0.05 else "NOT SIGNIFICANT ❌"
    print(f"{persona.upper()}: mean drift={persona_data.mean():.4f}, p-value={p_value:.4f} — {significance}")

# Heatmap
pivot = df.pivot_table(values='drift', index='topic', columns='persona')
plt.figure(figsize=(14, 10))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, linewidths=0.5)
plt.title('Bias Drift Heatmap: Topics vs Personas', fontsize=14, fontweight='bold')
plt.xlabel('Persona')
plt.ylabel('Topic')
plt.tight_layout()
plt.savefig('bias_heatmap.png', dpi=150)
print("\n✅ Heatmap saved as bias_heatmap.png")

# Average drift by persona
plt.figure(figsize=(10, 6))
avg_drift = df.groupby('persona')['drift'].mean()
colors = ['gray', 'blue', 'red', 'orange']
plt.bar(avg_drift.index, avg_drift.values, color=colors)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Average Sentiment Drift by Persona', fontsize=14, fontweight='bold')
plt.xlabel('Persona')
plt.ylabel('Average Drift from Original')
plt.tight_layout()
plt.savefig('bias_by_persona.png', dpi=150)
print("✅ Bar chart saved as bias_by_persona.png")

print("\n🔥 STUDY COMPLETE!")