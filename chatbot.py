from groq import Groq
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from project root regardless of where script is run from
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_metrics():
    """Load latest detection metrics"""
    try:
        # Path relative to project root
        metrics_path = Path(__file__).parent.parent / "metrics" / "eval_results.json"
        with open(metrics_path) as f:
            return json.load(f)
    except:
        return {}

def load_params():
    """Load project params"""
    try:
        import yaml
        params_path = Path(__file__).parent.parent / "params.yaml"
        with open(params_path) as f:
            return yaml.safe_load(f)
    except:
        return {}

def get_system_prompt():
    metrics = load_metrics()

    violation_events = metrics.get("violation_events", "N/A")
    violation_frames = metrics.get("violation_frames", "N/A")
    total_frames     = metrics.get("total_frames", "N/A")
    class_counts     = metrics.get("class_counts", {})
    source           = metrics.get("source", "N/A")
    timestamp        = metrics.get("timestamp", "N/A")

    class_counts_str = "\n".join(
        f"  - {cls}: {count}" for cls, count in class_counts.items()
    ) if class_counts else "  No data available"

    return f"""You are a Construction Site Safety AI Assistant for a PPE (Personal Protective Equipment) detection system.

You have access to the following REAL-TIME data from the latest detection run:

DETECTION SUMMARY:
- Source video: {source}
- Timestamp: {timestamp}
- Total frames analysed: {total_frames}
- Frames with violations: {violation_frames}
- Total violation events: {violation_events}

DETECTED CLASS COUNTS:
{class_counts_str}

PPE CLASSES IN THIS SYSTEM:
- Safety violations (critical): NO-Hardhat, NO-Mask, NO-Safety Vest
- Compliant PPE: Hardhat, Mask, Safety Vest, Safety Cone
- Other detections: Person, machinery, vehicle

CONFIDENCE THRESHOLDS:
- Violation classes: 0.35 (more sensitive to catch violations)
- Standard classes: 0.25

YOUR ROLE:
You help construction site safety managers by:
1. Answering questions about detection results and performance
2. Explaining which violations were detected and how many
3. Suggesting practical safety improvements for the site
4. Interpreting detection data in plain, non-technical language
5. Advising on PPE compliance strategies

Always prioritise worker safety. Keep answers concise and practical for on-site managers."""


def chat(message, history):
    """Send message to Groq and get response"""

    # Build messages list with system prompt + history
    messages = [{"role": "system", "content": get_system_prompt()}]

    for human, assistant in history:
        messages.append({"role": "user",      "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )

    return response.choices[0].message.content