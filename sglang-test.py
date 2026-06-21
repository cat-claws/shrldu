from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "user", "content": "Say hello in five words."}
    ],
    temperature=0,
    max_tokens=16,
)

print(response.choices[0].message.content)