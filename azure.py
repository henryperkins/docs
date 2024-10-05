import os
from openai import AzureOpenAI

# Load environment variables
endpoint = os.getenv("ENDPOINT_URL", "https://openai-eastus2-hp.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt4o")
search_endpoint = os.getenv("SEARCH_ENDPOINT", "https://searchaihp.search.windows.net/")
search_key = os.getenv("SEARCH_KEY", "your_search_key_here")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "your_azure_openai_key_here")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# Create a completion
try:
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            }
        ],
        past_messages=10,
        max_tokens=16384,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        extra_body={
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "filter": None,
                    "endpoint": search_endpoint,
                    "index_name": "azureblob-index",
                    "semantic_configuration": "",
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    },
                    "embedding_dependency": {
                        "type": "endpoint",
                        "endpoint": "https://openai-eastus2-hp.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-03-15-preview",
                        "authentication": {
                            "type": "api_key",
                            "key": "86685fea68ec463c941983f7f38b717a"
                        }
                    },
                    "query_type": "vector_simple_hybrid",
                    "in_scope": True,
                    "role_information": "You are an AI assistant that helps people find information.",
                    "strictness": 3,
                    "top_n_documents": 5
                }
            }]
        }
    )

    print(completion.to_json())

except Exception as e:
    print(f"Error creating completion: {e}")