import os
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAIError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Azure OpenAI credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt4o")

if not api_key or not endpoint:
    logger.error("Azure OpenAI credentials not found in environment variables.")
    raise ValueError("Please set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables.")

# Initialize the Azure OpenAI client
def get_azure_client():
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2023-05-15",
    )

# Test the Azure OpenAI integration
def test_azure_openai():
    client = get_azure_client()
    try:
        # Make a simple completion request
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        )
        
        # Print the response
        logger.info("Azure OpenAI Response:")
        logger.info(response.choices[0].message.content)
        logger.info("Test successful!")
        return True
    except OpenAIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_azure_openai()
    if success:
        logger.info("Azure OpenAI integration test passed.")
    else:
        logger.error("Azure OpenAI integration test failed.")
