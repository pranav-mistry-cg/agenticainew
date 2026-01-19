import asyncio
import os
import urllib.parse
from autogen_agentchat.agents import AssistantAgent
import feedparser
import logging
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from sqlalchemy import text
from autogen_core.models import UserMessage, SystemMessage
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class RSSResearchAgent:
    def __init__(self):
        self.OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT"),
        self.OPENAI_KEY = os.getenv("OPENAI_KEY"),
        self.API_VERSION = os.getenv("API_VERSION", "2025-01-01-preview")
        self.MODEL = os.getenv("MODEL", "gpt-4.1")

    async def run_agent(self, search_result: str):
        agent_instance = self.agent()
        message_to_send = TextMessage(content=search_result, source="user")
        logger.info("Sending message to agent...")
        response = await agent_instance.on_messages(
            [message_to_send], cancellation_token=CancellationToken()
        )
        return response.chat_message.content

    def get_openai_client(self):
        return AzureOpenAIChatCompletionClient(
            api_key=self.OPENAI_KEY,
            azure_deployment=self.MODEL,
            api_version=self.API_VERSION,
            azure_endpoint=self.OPENAI_ENDPOINT,
            model=self.MODEL,
            temperature=0.3,
        )

    def agent(self):
        return AssistantAgent(
            name="GoogleRSS_agent",
            model_client=self.get_openai_client(),
            description="An agent that aggregate  the GoogleRSS agent results.",
            system_message=(
                """
                   Summarize the following news in 3 crisp business-friendly bulleted sentences.
                    """
            ),
            reflect_on_tool_use=True,
        )


async def run_rss_research_agent(input_data):
    agent = RSSResearchAgent()
    response = await agent.run_agent(input_data)

    return response


class RssFeedNewsProcessor(ABC):
    @abstractmethod
    def build_rss_feed_url(self, company, categories) -> str:
        """Builds a News RSS feed URL based on the company name and categories."""

    def fetch_rss_entries(self, rss_url):
        feed = feedparser.parse(rss_url)

        return feed.entries

    def parse_news_entries(self, entries):
        news_items = []

        for entry in entries:
            news_items.append(
                {
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.published if "published" in entry else "",
                    "summary": entry.summary if "summary" in entry else "",
                    "source": entry.source.title if "source" in entry else "Unknown",
                }
            )

        return news_items[0:10]  # Limit to first 2 items for testing

    def classify_article(self, text):
        t = text.lower()

        if any(
            k in t for k in ["ceo", "cfo", "cto", "chairman", "appointed", "resigns"]
        ):
            print("Classified as Leadership")
            return "Leadership"

        if any(
            k in t
            for k in [
                "strategy",
                "vision",
                "expansion",
                "roadmap",
                "acquisition",
                "merger",
            ]
        ):
            print("Classified as Strategy")
            return "Strategy"

        if any(
            k in t
            for k in ["regulation", "policy", "compliance", "directive", "guideline"]
        ):
            print("Classified as Regulatory")
            return "Regulatory"

        print("Classified as Regulatory")
        return "Other"


class GoogleRssNewsProcessor(RssFeedNewsProcessor):

    def build_rss_feed_url(self, company, categories):
        # Example categories: ["leadership", "strategy", "regulation"]
        query = company + " " + " OR ".join(categories)
        encoded_query = urllib.parse.quote(query)

        # Google News RSS endpoint
        url = (
            f"https://news.google.com/rss/search?q={encoded_query}"
            f"&hl=en-IN&gl=IN&ceid=IN:en"
        )

        return url


@dataclass
class NewsProcessResult:
    RssProcessor: RssFeedNewsProcessor

    async def process_company(self, company, categories):

        url = self.RssProcessor.build_rss_feed_url(company, categories)
        entries = self.RssProcessor.fetch_rss_entries(url)
        parsed = self.RssProcessor.parse_news_entries(entries)

        results = []

        for item in parsed:
            combined_text = f"{item['title']}\n{item['summary']}"
            category = self.RssProcessor.classify_article(combined_text)

            summary = await run_rss_research_agent(combined_text)

            results.append(
                {
                    "company": company,
                    "title": item["title"],
                    "url": item["link"],
                    "published": item["published"],
                    "source": item["source"],
                    "category": category,
                    "summary": summary,
                }
            )

        return results


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run Google News RSS for a company.")
    parser.add_argument(
        "--company",
        required=False,
        default="Post NL",
        help="Company name, e.g., 'Microsoft'",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=False,
        default=["AI", "press", "strategy"],
        help="Categories list, e.g., AI cloud strategy",
    )
    parser.add_argument("--limit", type=int, default=5, help="Max items to print")
    args = parser.parse_args()

    newsProcessor = NewsProcessResult(RssProcessor=GoogleRssNewsProcessor())
    results = asyncio.run(newsProcessor.process_company(args.company, args.categories))
    print(json.dumps(results[: args.limit], indent=2))
