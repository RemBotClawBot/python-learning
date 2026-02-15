"""
web_scraping.py

Practical examples of gathering data from the web using requests + BeautifulSoup.
Demonstrates:
- Sending HTTP requests with custom headers
- Parsing HTML documents for structured data
- Cleaning and exporting scraped content
- Respectful scraping practices with delays & guards
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Python-Learning-Bot/1.0 (+https://github.com/openclaw)"
BASE_URL = "https://quotes.toscrape.com"

# Offline fallback HTML snippet (for environments without internet)
FALLBACK_HTML = """
<html>
  <body>
    <div class="quote">
      <span class="text">“The world as we have created it is a process of our thinking.”</span>
      <small class="author">Albert Einstein</small>
      <div class="tags">
        <a class="tag">change</a>
        <a class="tag">deep-thoughts</a>
      </div>
    </div>
    <div class="quote">
      <span class="text">“It is our choices, Harry, that show what we truly are.”</span>
      <small class="author">J.K. Rowling</small>
      <div class="tags">
        <a class="tag">choices</a>
        <a class="tag">inspirational</a>
      </div>
    </div>
  </body>
</html>
"""


@dataclass
class Quote:
    text: str
    author: str
    tags: List[str]

    def to_dict(self) -> dict:
        return {"text": self.text, "author": self.author, "tags": self.tags}


def fetch_page(url: str, timeout: int = 10) -> str:
    """Fetch HTML from a given URL with basic error handling."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_quotes(html: str) -> List[Quote]:
    """Parse quote blocks from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    quotes: List[Quote] = []

    for block in soup.select("div.quote"):
        text = block.select_one("span.text")
        author = block.select_one("small.author")
        tags = [tag.text.strip() for tag in block.select("div.tags a.tag")]

        if text and author:
            quotes.append(Quote(text=text.text.strip("“”"), author=author.text.strip(), tags=tags))

    return quotes


def scrape_quotes(pages: int = 2, delay: float = 1.0) -> List[Quote]:
    """Scrape multiple pages of the website with throttling."""
    all_quotes: List[Quote] = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/page/{page}"
        print(f"Fetching {url}")
        try:
            html = fetch_page(url)
        except Exception as exc:
            print(f"  ⚠️  Network error ({exc}). Falling back to embedded HTML sample.")
            html = FALLBACK_HTML

        quotes = parse_quotes(html)
        print(f"  ✓ Extracted {len(quotes)} quotes")
        all_quotes.extend(quotes)

        # Respectful scraping – small pause between requests
        time.sleep(delay)

    return all_quotes


def save_quotes(quotes: List[Quote], filename: str = "quotes.json") -> None:
    """Persist scraped data for later analysis."""
    payload = [quote.to_dict() for quote in quotes]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📁 Saved {len(quotes)} quotes to {filename}")


def display_summary(quotes: List[Quote]) -> None:
    """Print highlight statistics to the console."""
    if not quotes:
        print("No quotes collected.")
        return

    authors = {}
    tag_frequency = {}

    for quote in quotes:
        authors[quote.author] = authors.get(quote.author, 0) + 1
        for tag in quote.tags:
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

    popular_author = max(authors, key=authors.get)
    popular_tag = max(tag_frequency, key=tag_frequency.get)

    print("\nSUMMARY")
    print("-" * 40)
    print(f"Total quotes: {len(quotes)}")
    print(f"Unique authors: {len(authors)}")
    print(f"Most quoted author: {popular_author} ({authors[popular_author]} quotes)")
    print(f"Most popular tag: #{popular_tag} ({tag_frequency[popular_tag]} uses)")

    print("\nSample Quotes:")
    for quote in quotes[:3]:
        tags = ", ".join(quote.tags) or "no tags"
        print(f"  • \"{quote.text[:60]}...\" — {quote.author} [{tags}]")


if __name__ == "__main__":
    print("=" * 60)
    print("WEB SCRAPING DEMO")
    print("=" * 60)
    print("Target website: quotes.toscrape.com (built for practice)")
    print("Always check robots.txt before scraping a new site!\n")

    quotes = scrape_quotes(pages=2, delay=0.5)
    display_summary(quotes)
    save_quotes(quotes)

    print("\nNext steps:")
    print("  • Schedule regular scrapes and append to a datastore")
    print("  • Combine with data_analysis.py for tag trends")
    print("  • Handle pagination automatically when Next button exists")
    print("  • Implement caching to avoid duplicate requests")

    print("\n✅ Web scraping workflow complete")
