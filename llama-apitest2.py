from pathlib import Path
from openai import OpenAI
import json
from datetime import datetime


API_KEY = Path("mckinney-api.key").read_text().strip()
client = OpenAI(base_url="https://genai.rcac.purdue.edu/api", api_key=API_KEY)


def content_to_text(content):
    """Flatten text whether server returns a string or list of parts."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content

def queryllama(prompts: list):
    messages = []
    replies = []
    for i, prompt in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model="llama4:latest",
            messages=messages,
        )
        reply = resp.choices[0].message
        reply_text = content_to_text(reply.content)
        replies.append(reply_text)
        # print(f"Response {i}: {reply_text}")
        # Save assistant reply to maintain context for the next turn.
        messages.append({"role": reply.role, "content": reply.content})
    return replies



with open("gender_bias_tests_small.json", "r") as test_dataset:
    entries = json.load(test_dataset)


start_time = datetime.now()
print(f"Starting testing at: {start_time}")

test_results = []
i = 0
for item in entries:
    prompts = []
    i += 1
    testid = item["id"]
    print(f"({datetime.now()}) Running test {i} of {len(entries)}: {testid}")
    

    role = item["role"]
    preprompt = item["preprompt"]
    prompt= item["prompt"]
    followup = item["followup"]
    test_id = item["id"]
    category = item["category"]
    age = item["age"]
    gender = item["gender"]

    # Set up prompts to be passed to AI
    if preprompt != "none":
        prompts.append(preprompt)
    prompts.append(prompt)
    prompts.append(followup)
    
    # Pass prompts to AI
    responses = queryllama(prompts=prompts)
    # Responses is a list of strings, one for each reply
    if preprompt == "none":
        preprompt_response = "none"
        prompt_response = responses[0]
        followup_response = responses[1]
    else:
        preprompt_response = responses[0]
        prompt_response = responses[1]
        followup_response = responses[2]


    test_results.append({
                    "id": test_id,
                    "category": category,
                    "age": age,
                    "gender": gender,
                    "role": role,
                    "preprompt": preprompt,
                    "preprompt-response": preprompt_response,
                    "prompt": prompt,
                    "prompt-response": prompt_response,
                    "followup": followup,
                    "followup-response": followup_response
                })

out_path = Path(__file__).with_name("llama4results.json")
out_path.write_text(
    json.dumps(test_results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Wrote {len(test_results)} test cases to:\n  {out_path.resolve()}")
end_time = datetime.now()
print(f"Testing ended at: {end_time}")
print(f"Testing took: {end_time - start_time}")

