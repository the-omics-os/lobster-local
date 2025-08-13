import json
import boto3

# Replace with the specific Claude model ID you have access to
MODEL_ID = 'us.anthropic.claude-opus-4-20250514-v1:0'

bedrock_runtime = boto3.client("bedrock-runtime")

# Example prompt
prompt = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a short poem about a cat."}]
        }
    ]
}

response = bedrock_runtime.invoke_model(
    body=json.dumps(prompt),
    modelId=MODEL_ID,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
print(response_body["content"][0]["text"])
