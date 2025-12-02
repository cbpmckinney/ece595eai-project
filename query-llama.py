import json
import requests
from pathlib import Path


url = "https://genai.rcac.purdue.edu/api/chat/completions"

with open("mckinney-api.key", "r") as f:
    api_key = f.read().strip()


def getllamaresponse(prompt_content: str) -> str:
    # Use a streaming response so requests does not try to buffer the whole chunked payload.

    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
    }
    body = {
    "model": "llama4:latest",
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
            # Server returns SSE-style "data: ..." lines; strip the prefix when present.
            if line.startswith("data: "):
                line = line[len("data: "):]
            if line.strip() == "[DONE]":
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue
            #print(json.dumps(payload, ensure_ascii=False))
            response_content = (
                payload.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
                )
            response_text += response_content

            
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
        preprompt_response_text = getllamaresponse(prompt_content=preprompt)
    else:
        preprompt_response_text = "none"

    response_text = getllamaresponse(prompt_content=prompt)
    followup_text = getllamaresponse("Why?")
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

out_path = Path(__file__).with_name("llama4results.json")
out_path.write_text(
    json.dumps(test_results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Wrote {len(test_results)} test cases to:\n  {out_path.resolve()}")

