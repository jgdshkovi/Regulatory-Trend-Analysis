import feedparser
import pandas as pd

def parse_rss_feeds(feed_links):
    feed_data = []
    for link in feed_links:
        feed = feedparser.parse(link)
        for entry in feed.entries:
            feed_data.append({
                'feed_link': link,
                'title': entry.get('title', ''),
                'published': entry.get('published', ''),
                'summary': entry.get('summary', ''),
                'link': entry.get('link', '')
            })
    return feed_data

def main():
    # Input list of RSS feed links
    rss_feed_links = [
        'https://www.justice.gov/news/rss?m=1'
        # 'https://example.com/feed2',
        # Add more feed links as needed
    ]

    # Parse RSS feeds
    feed_data = parse_rss_feeds(rss_feed_links)

    # Convert data to DataFrame
    df = pd.DataFrame(feed_data)

    # Display DataFrame
    print(df)

if __name__ == "__main__":
    main()
