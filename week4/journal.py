from openai import OpenAI
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

# Config Flag: Change this value to switch modes
USE_LOCAL = True  # True = Ollama (Offline/Privacy), False = Groq API (Online/High Quality)

# Initialize the client based on the selection
if USE_LOCAL:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "llama3.1:8b"
else:
    # Read API key from environment variable (loaded from .env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ OPENAI_API_KEY not found! Please create a .env file with your API key."
        )
    
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=api_key
    )

    MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")

console = Console()

def analyze_journal(entry_text: str) -> tuple:
    """
    Function that takes the daily journal entry and returns:
    - AI analysis (mood, summary, advice)
    - Token usage and performance metrics (time & speed)
    """
    start_time = time.time()
    
    system_prompt = """You are an empathetic, insightful, and supportive personal journaling assistant. 
 The user will share their daily thoughts. You must respond in a structured format using Markdown:
 1. **Mood Analysis:** Briefly analyze their emotional state today.
 2. **Daily Summary:** A short, poetic, or inspiring summary of their day.
 3. **Advice/Reflection:** One thoughtful piece of advice or a reflective question for tomorrow.
 Keep the tone warm, private, and encouraging."""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Today I want to journal about:\n\n{entry_text}"}
        ],
        temperature=0.8  # Higher temperature for more creative and empathetic writing
    )
    
    end_time = time.time()
    result = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    time_taken = end_time - start_time
    
    # Calculate generation speed
    speed = tokens_used / time_taken if time_taken > 0 else 0
    
    return result, tokens_used, time_taken, speed

def main():
    mode = "🔒 LOCAL (Offline & Private)" if USE_LOCAL else "☁️ HOSTED (Groq API)"
    console.print(Panel(
        f"[bold green]Welcome to your Private Journaling Assistant[/bold green]\n"
        f"Running in {mode} mode using {MODEL_NAME}",
        title="Journal AI"
    ))
    
    console.print("\n[dim]Write about your day. When you're done, press Enter twice (or type 'DONE' on a new line).[/dim]")
    
    # Collect user input
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'DONE' or line == '':
                if lines: 
                    break
            else:
                lines.append(line)
        except EOFError:
            break
    
    user_entry = "\n".join(lines)
    
    if not user_entry.strip():
        console.print("[red]You didn't write anything! Exiting.[/red]")
        return
    
    console.print("\n[yellow]Analyzing your thoughts...[/yellow]")
    
    # Call the model
    analysis, tokens, time_sec, speed = analyze_journal(user_entry)
    
    # Display the result beautifully
    console.print(Panel(Markdown(analysis), title="📖 Your Journal Reflection", border_style="blue"))
    
    # Display statistics (important for the Benchmark)
    console.print(f"\n[dim]⚙️ Stats: {tokens} tokens | {time_sec:.2f} seconds | {speed:.2f} tokens/sec[/dim]")

if __name__ == "__main__":
    main()