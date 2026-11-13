%cd /content/drive/MyDrive/project/marketmind

from pathlib import Path

p = Path("commentary.py")
code = p.read_text()

start = code.index("def generate(facts):")
end = code.index('\n\nif __name__ == "__main__":', start)

replacement = '''def generate(facts):
    src = facts_text(facts)

    # Use Anthropic when an API key is configured.
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        prompt = (
            "Write a 3-4 sentence market comment for a trading desk. "
            "Use ONLY the facts below. Do not invent numbers, causes or events. "
            "Say plainly that sentiment is a noisy signal.\\\\n\\\\n" + src
        )

        msg = anthropic.Anthropic().messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        output = msg.content[0].text
        return output, check_numbers(output, src), "Anthropic LLM"

    # Deterministic fallback: no API call and no invented market facts.
    change = facts["djia_daily_change_pct"]
    sentiment = facts["mean_sentiment"]

    direction = "rose" if change > 0 else "fell" if change < 0 else "was unchanged"
    sentiment_view = (
        "positive" if sentiment > 0.05
        else "negative" if sentiment < -0.05
        else "approximately neutral"
    )

    output = (
        f"On {facts['date']}, the DJIA {direction} by {abs(change):.2f}%. "
        f"Mean FinBERT sentiment was {sentiment:.3f}, indicating "
        f"{sentiment_view} tone across {facts['n_headlines']} scored headlines. "
        "This is a historical observation, not evidence of causation or a "
        "reliable trading signal. Financial-news sentiment is noisy and should "
        "not be used alone to make investment decisions."
    )

    return output, check_numbers(output, src), "Deterministic fallback"
'''

p.write_text(code[:start] + replacement + code[end:])
print("Fallback added successfully.")