from openai import OpenAI

client = OpenAI(
    base_url=ENDPOINT,
    api_key=API_KEY
)

completion = client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message.content)