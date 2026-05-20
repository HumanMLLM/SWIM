# !export API_KEY=api_key
# !pip install google-generativeai
import os
from tqdm import tqdm
import time
import requests
import PIL.Image
import json
import base64
from tqdm import tqdm
import argparse
from openai import AzureOpenAI
import requests
import time

# def init():
#     client = AzureOpenAI(
#         azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
#         api_key=os.getenv("AZURE_OPENAI_KEY"),  
#         api_version="2024-02-15-preview"
#     )

#     return client

# def interaction(client, message_text):
#     completion = client.chat.completions.create(
#         model=os.getenv("AZURE_OPENAI_DEPLOYNAME"),
#         messages = message_text,
#         temperature=0.7,
#         max_tokens=800,
#         top_p=0.95,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None
#     )

#     return completion


def main(args):
    # client = init()

    API_URL = os.environ["AZURE_OPENAI_ENDPOINT"]
    API_KEY = os.environ["AZURE_OPENAI_KEY"]
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = []
    for line in open(args.input_file):
        d = json.loads(line)
        data.append(d)

    with open('./data/VideoRefer/videorefer/eval/videorefer_bench_d/system.txt', 'r') as f:
        system_message = f.read()

    for d in tqdm(data):
        if 'pred' not in d:
            continue

        gt = '##Correct answer: '+d['caption'] + '\n'
        pred = '##Predicted answer: '+d['pred'] +'\n'
    
        messages = [
            {"role": "system", "content":[{"type": "text", "text": system_message}]},
            {"role": "user", "content":[{"type": "text", "text": gt+pred}]}
        ]

        payload = {
        "model": os.environ["AZURE_OPENAI_DEPLOYNAME"],
        "messages": messages,
        "temperature": 0,
    }
        for i in range(20):
            try:
                # completion = interaction(client, message)

                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

                response.raise_for_status()  # Raises HTTPError for bad responses
                try:
                    response_data = response.json()  # Attempt to parse JSON
                except requests.exceptions.JSONDecodeError:
                    eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                    continue  # Skip to next retry
                generate_content = response_data["choices"][0]["message"]["content"].strip()
                if generate_content != "":
                    # return content, response_data["model"]
                    print(generate_content)
                # generate_content = completion.choices[0].message.content
                d['gpt'] = generate_content
                break

            except Exception as e:
                print("error. model generation failed.")
                time.sleep(1)
                

        b = json.dumps(data)
        f2 = open(args.output_file, 'w')
        f2.write(b)
        f2.close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-4o")
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument("--api-key", required=True, type=str, help="Azure Openai API key.")
    parser.add_argument("--api-endpoint", required=True, type=str, help="Azure Openai API endpoint.")
    parser.add_argument("--api-deployname", required=True, type=str, help="Azure Openai API deployname.")
    args = parser.parse_args()

    # Set the OpenAI API key.
    os.environ["AZURE_OPENAI_KEY"] = args.api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = args.api_endpoint
    os.environ["AZURE_OPENAI_DEPLOYNAME"] = args.api_deployname

    # client = init()

    main(args)
