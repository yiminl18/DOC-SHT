import os
from openai import AzureOpenAI

api_key_path = ''
ENDPOINT_URL = ''
api_version_name = ''

def gpt_4o_azure(input, 
                            key_path=api_key_path,
                            max_tokens=800,
                            temperature=0):
    """
    Get response from Azure OpenAI API.
    
    Args:
        prompt (str): The text prompt to send to the model
        key_path (str): Path to the API key file
        max_tokens (int): Maximum tokens for response
        temperature (float): Response randomness (0-1)
        
    Returns:
        str: The response content from the model
    """
    prompt = input[0] + input[1] 
    # Read API key
    with open(key_path, 'r') as f:
        api_key = f.read().strip()
    
    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=os.getenv("ENDPOINT_URL", ENDPOINT_URL),
        api_key=api_key,
        api_version=api_version_name,
    )
    
    # Generate response
    completion = client.chat.completions.create(
        model=os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stream=False
    )
    
    return completion.choices[0].message.content

