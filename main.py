import pandas as pd
from transformers import pipeline

df = pd.read_csv("NepalPolitics - Sheet1.csv")
df['event'] = df['event'].fillna('')
df['year'] = df['year'].fillna('Unknown')

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_events(query):
    if query.isdigit():
        subset = df[df['year'].astype(str) == query]
    else:
        subset = df[df['event'].str.contains(query, case=False, na=False)]

    if subset.empty:
        return f"❌ No events found for '{query}'."

    text = " ".join(subset['event'].tolist())[:1000]  
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    
    return f"\n📝 Summary for '{query}':\n{summary}\n"
if __name__ == "__main__":
    print("🧠 Nepal Political Events Summarizer (type 'exit' to quit)\n")
    while True:
        user_input = input("🔍 Enter a year or keyword: ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break
        print(summarize_events(user_input))
