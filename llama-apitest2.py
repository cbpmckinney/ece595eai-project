from pathlib import Path
from openai import OpenAI

API_KEY = Path("mckinney-api.key").read_text().strip()
client = OpenAI(base_url="https://genai.rcac.purdue.edu/api", api_key=API_KEY)


def content_to_text(content):
    """Flatten text whether server returns a string or list of parts."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content


# Keep a running conversation so each follow-up has previous context.
messages = []
prompts = [
    "Pick an example animal from the list: dog, cat, monkey, bird.  Tell me which you chose.",
    "How many legs does that animal have?",
    "Tell me an interesting fact about whales.",
]

for i, prompt in enumerate(prompts, start=1):
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model="llama4:latest",
        messages=messages,
    )
    reply = resp.choices[0].message
    reply_text = content_to_text(reply.content)
    print(f"Response {i}: {reply_text}")
    # Save assistant reply to maintain context for the next turn.
    messages.append({"role": reply.role, "content": reply.content})
