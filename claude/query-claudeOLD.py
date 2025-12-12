import json
import requests
from pathlib import Path


url = "https://api.anthropic.com/v1/messages"

with open("claude-api.key", "r") as f:
    api_key = f.read().strip()


def getclauderesponse(prompt_content: str) -> str:
    # Use a streaming response so requests does not try to buffer the whole chunked payload.

    headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
    }
    body = {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 4096,
    "messages": [
    {
        "role": "user",
        "content": prompt_content,
    }
    ],
    "stream": True
    }
    response = requests.post(url, headers=headers, json=body, stream=True, timeout=120)
    response_text = ""
    try:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            # Skip keepalives / empty lines.
            if not line:
                continue
            # Anthropic API returns SSE-style "event:" and "data:" lines
            if line.startswith("event: "):
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            else:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Handle content_block_delta events which contain the actual text
            if payload.get("type") == "content_block_delta":
                delta = payload.get("delta", {})
                if delta.get("type") == "text_delta":
                    response_text += delta.get("text", "")
            elif payload.get("type") == "message_stop":
                break


    except requests.exceptions.ChunkedEncodingError as exc:
        # If the server closes the connection early, show what we received so far.
        raise Exception(f"Stream ended prematurely: {exc}") from exc
    return response_text



with open("gender_bias_tests_small.json", "r") as test_dataset:
    entries = json.load(test_dataset)

test_results = []

for item in entries:
    role = item["role"]
    preprompt = item["preprompt"]
    prompt= item["prompt"]

    if preprompt != "none":
        preprompt_response_text = getclauderesponse(prompt_content=preprompt)
    else:
        preprompt_response_text = "none"

    response_text = getclauderesponse(prompt_content=prompt)
    followup_text = getclauderesponse("Why?")
    test_id = item["id"]
    category = item["category"]
    age = item["age"]
    gender = item["gender"]


    test_results.append({
                    "id": test_id,
                    "category": category,
                    "age": age,
                    "gender": gender,
                    "role": role,
                    "preprompt": preprompt,
                    "prompt": prompt,
                    "response": response_text,
                    "followup": followup_text
                })

out_path = Path(__file__).with_name("claude_haiku_results.json")
out_path.write_text(
    json.dumps(test_results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Wrote {len(test_results)} test cases to:\n  {out_path.resolve()}")

