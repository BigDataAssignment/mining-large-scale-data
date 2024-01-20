"""
Mental health subreddits scrapping spider.

Example:
    >>> python ./reddit_spider.py

Author:
    Adam Darmanin

Notes:
    You need to have a .ENV file with the following settings:
    CLIENT_ID=YOUR_CLIENT_ID
    CLIENT_SECRET=YOUR_SECRET
    USER_AGENT=uom_research v1.0 by /u/YOUR_USER_NAME
    SUBREDDITS=MentalHealth,depression,anxiety
    See: https://www.reddit.com/wiki/api/
"""
import praw
import threading
import csv
from datetime import datetime, timezone
import dateutil.parser
from dotenv import load_dotenv
import os
import time
import logging


def scrape_subreddit(subreddit_name, until_date, post_limit=100):
    """
    Scrape posts from a specified subreddit up to a certain date and save them to a CSV file.
    NB: I gave it 5 exponential retries if it fails to scrape.

    Args:
        subreddit_name (str): The name of the subreddit to scrape.
        post_limit (int): The maximum number of posts to scrape.
        until_date (datetime): The cutoff date for scraping posts.

    Example:
        scrape_subreddit('MentalHealth', 10, datetime(2023, 1, 1))
        This will scrape posts from the 'MentalHealth' subreddit up to January 1, 2023.
    """
    retry_attempts = 0
    max_retries = 5
    wait_time = 10
    while retry_attempts < max_retries:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            oldest_post_date = None

            data_for_csv = []
            for post in subreddit.hot(limit=post_limit):
                post_date = datetime.fromtimestamp(post.created_utc, timezone.utc)
                if post_date < until_date:
                    break

                data = {
                    'title': post.title,
                    'selftext': post.selftext,
                    'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'over_18': post.over_18,
                    'subreddit': post.subreddit.display_name
                }
                data_for_csv.append(data)

                if oldest_post_date is None or post.created_utc < oldest_post_date:
                    oldest_post_date = post.created_utc

            if oldest_post_date is None:
                logging.error(f"Got none oldest post date, no posts scraped for: {subreddit_name}")
                return

            today_date = datetime.today().strftime('%Y-%m-%d')
            oldest_post_date_formatted = datetime.fromtimestamp(oldest_post_date).strftime('%Y-%m-%d')
            filename = f"./data/{today_date}_to_{oldest_post_date_formatted}_{subreddit_name}.csv"
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=data_for_csv[0].keys())
                writer.writeheader()
                writer.writerows(data_for_csv)

            logging.info(f"Data from {subreddit_name} saved to {filename}")

        except praw.exceptions.APIException as e:
            logging.error(f"API Exception for {subreddit_name}: {e}")
            time.sleep(wait_time)
            retry_attempts += 1
            wait_time *= 2

        if retry_attempts == max_retries:
            logging.error(f"Failed to scrape {subreddit_name} after {max_retries} attempts.")

def main():
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent = os.getenv('USER_AGENT')
    until_date_str = os.getenv('SCRAPE_UNTIL_DATE', '2023-01-01')
    until_date = dateutil.parser.parse(until_date_str).replace(tzinfo=timezone.utc)
    subreddits = os.getenv('SUBREDDITS').split(',')
    max_posts = int(os.getenv('MAX_POSTS', 100))

    global reddit
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=60,
        check_for_async=False
    )

    threads = []
    for subreddit_name in subreddits:
        thread = threading.Thread(target=scrape_subreddit, args=(subreddit_name, until_date, max_posts))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    logging.info("Scraping complete.")

if __name__ == "__main__":
    def _config_logging():
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        for logger_name in ("praw", "prawcore"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

    load_dotenv()
    _config_logging()

    main()
