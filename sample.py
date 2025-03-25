import requests
import json
# from sseclient import SSEClient

def test_streaming_api():
    url = "https://fdluc8lct2ma63-8000.proxy.runpod.net/api/inference/"
    
    # Request payload with stream=True
    payload = {
        "prompt": "You are an expert radiographer. Describe accurately what you see in this image.",
        "max_new_tokens": 500,
        "temperature": 0.7,
        "min_p": 0.05,
        "model": "RadiGenius",  # Replace with your actual model name
        "stream": True,
        "attachments": [
            "https://cdn.stocken.ir/media/post/fd4653bf-fda7-4804-a3d3-d3376a3abfd3/test.webp"
        ]  # Optional - Include if your API expects this field
    }
    
    # Method 1: Using requests with streaming
    print("Method 1: Using requests streaming")
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    # Remove 'data: ' prefix if using SSE format
                    data = line.decode('utf-8')
                    if data.startswith('data: '):
                        data = data[6:]
                    print(data, end='', flush=True)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    # # Method 2: Using SSEClient (better for SSE streams)
    # print("\n\nMethod 2: Using SSEClient")
    # try:
    #     response = requests.post(url, json=payload, stream=True)
    #     client = SSEClient(response)
    #     for event in client.events():
    #         print(event.data, end='', flush=True)
    # except Exception as e:
    #     print(f"Error: {e}")

if __name__ == "__main__":
    test_streaming_api()