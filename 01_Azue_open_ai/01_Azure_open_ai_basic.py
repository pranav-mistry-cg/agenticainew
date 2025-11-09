import asyncio
import os
from dotenv import load_dotenv
from autogen_core.models import UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

async def main():
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