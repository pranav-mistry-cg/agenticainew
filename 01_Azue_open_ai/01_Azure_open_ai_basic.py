import asyncio
import os
from dotenv import load_dotenv
from autogen_core.models import UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

async def main():
    """
    Asynchronously initializes environment variables, prints Azure OpenAI configuration values, 
    creates an AzureOpenAIChatCompletionClient, sends a user message to the model, and prints the model's response.
    Steps performed:
    1. Loads environment variables from a .env file.
    2. Prints endpoint, API key, and deployment name for debugging.
    3. Initializes the AzureOpenAIChatCompletionClient with configuration from environment variables.
    4. Sends a prompt to the model requesting three bullet points on why RAG improves answer accuracy.
    5. Prints the model's response.
    Returns:
        None
    """
    load_dotenv(override=True)
    print(os.getenv("OPENAI_ENDPOINT"))
    print(os.getenv("OPENAI_KEY"))
    print(os.getenv("AZURE_OPENAI_DEPLOYMENT"))

    # You can pass endpoint/key explicitly, or rely on env vars
    client = AzureOpenAIChatCompletionClient(
        # model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-prod"),  # <-- deployment name
        model=os.getenv("MODEL", "gpt-4.1"),  # <-- deployment name
        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_KEY"),
        temperature=0.2,
        api_version=os.getenv("API_VERSION", "2025-01-01-preview")
    )

    result = await client.create([
        UserMessage(content="Give me three bullet points on why RAG improves answer accuracy.", source="user")
    ])

    # result.content is a string with the LLM answer
    print(result.content)

if __name__ == "__main__":
    asyncio.run(main())