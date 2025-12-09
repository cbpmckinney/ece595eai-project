import json
import requests
import time
from pathlib import Path
from datetime import datetime


url = "https://api.anthropic.com/v1/messages"

with open("claude-api.key", "r") as f:
    api_key = f.read().strip()


def getclauderesponse(messages: list) -> str:
    # Use a streaming response so requests does not try to buffer the whole chunked payload.

    headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
    }
    body = {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 4096,
    "messages": messages,
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

print(f"Starting tests for {len(entries)} test cases...")
start_time = time.time()

for item in entries:
    role = item["role"]
    preprompt = item["preprompt"]
    prompt= item["prompt"]

    # Build conversation with preprompt if it exists
    messages = []
    if preprompt != "none":
        messages.append({"role": "user", "content": preprompt})
        preprompt_response_text = getclauderesponse(messages=messages)
        messages.append({"role": "assistant", "content": preprompt_response_text})
    else:
        preprompt_response_text = "none"

    # Send main prompt in same context (if preprompt exists)
    messages.append({"role": "user", "content": prompt})
    response_text = getclauderesponse(messages=messages)

    # Add assistant's response to conversation and ask followup question
    messages.append({"role": "assistant", "content": response_text})
    messages.append({"role": "user", "content": "Why?"})
    followup_text = getclauderesponse(messages=messages)
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

end_time = time.time()
elapsed_time = end_time - start_time

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = results_dir / f"claude_haiku_{timestamp}.json"

out_path.write_text(
    json.dumps(test_results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print(f"\nCompleted {len(test_results)} test cases in {minutes}m {seconds:.2f}s")
print(f"Results saved to:\n  {out_path.resolve()}")

