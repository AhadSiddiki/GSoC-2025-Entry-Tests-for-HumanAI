import re
import json
import csv
import os
import datetime
import argparse
import time
import praw
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from dotenv import load_dotenv
from prawcore.exceptions import RequestException
from collections import Counter
import pandas as pd
# Load environment variables from .env file
load_dotenv()

# Create T1_output directory if it doesn't exist
output_dir = 'T1_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')

# Mental health keywords for filtering
MENTAL_HEALTH_KEYWORDS = [
    "depressed", "depression", "anxiety", "anxious", "overwhelmed",
    "suicidal", "suicide", "self-harm", "addiction", "substance abuse",
    "mental health", "therapy", "panic attack", "bipolar", "ptsd"
]


class RedditExtractor:
    def __init__(self, output_format='csv'):
        self.output_format = output_format.lower()
        self.data = []
        self.subreddit_stats = {}  # Add dictionary to track subreddit stats

        # Initialize Reddit API with increased timeout
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'Mental Health Data Extraction Script v1.0'),
            timeout=30  # Increased timeout from default 16 seconds
        )

    def extract_data(self, query_limit=100, subreddits=None, time_period='week', max_retries=3):
        """Extract posts from Reddit using mental health keywords"""
        print("Extracting data from Reddit...")

        # Default subreddits related to mental health if none specified
        if subreddits is None:
            subreddits = [
                "depression", "anxiety", "mentalhealth", "SuicideWatch",
                "addiction", "bipolar", "ptsd", "BPD", "schizophrenia"
            ]

        # Initialize statistics for each subreddit
        for subreddit in subreddits:
            self.subreddit_stats[subreddit] = {
                'total_posts_found': 0,
                'relevant_posts': 0,
                'total_likes': 0,
                'total_comments': 0,
                'keywords_found': Counter()
            }

        # Extract posts for each keyword from each subreddit
        for subreddit_name in subreddits:
            print(f"Searching in r/{subreddit_name}")

            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts = self._get_posts_with_retry(
                    subreddit=subreddit,
                    time_period=time_period,
                    query_limit=query_limit,
                    max_retries=max_retries
                )

                # Update total posts found
                self.subreddit_stats[subreddit_name]['total_posts_found'] = len(posts)

                # Process each post
                for post in posts:
                    # Check if post content contains any of our keywords
                    if self._is_relevant(post):
                        processed_post = self._process_post(post)
                        self.data.append(processed_post)
                        
                        # Update subreddit statistics
                        self.subreddit_stats[subreddit_name]['relevant_posts'] += 1
                        self.subreddit_stats[subreddit_name]['total_likes'] += post.score
                        self.subreddit_stats[subreddit_name]['total_comments'] += post.num_comments
                        
                        # Count keywords found
                        content = (post.title + " " + post.selftext).lower()
                        for keyword in MENTAL_HEALTH_KEYWORDS:
                            if keyword.lower() in content:
                                self.subreddit_stats[subreddit_name]['keywords_found'][keyword] += 1

                time.sleep(5)  # Rate limiting

            except Exception as e:
                print(f"Error processing subreddit r/{subreddit_name}: {str(e)}")
                time.sleep(10)
                continue

        # Save subreddit statistics
        self._save_subreddit_stats()
        
        print(f"Extracted {len(self.data)} posts in total")
        return self.data

    def _get_posts_with_retry(self, subreddit, time_period, query_limit, max_retries=3):
        """Get posts with retry logic for handling timeouts"""
        posts = []

        for attempt in range(max_retries):
            try:
                # Get posts from subreddit based on time period using keyword arguments
                if time_period == 'day':
                    posts = list(subreddit.top(time_filter='day', limit=query_limit))
                elif time_period == 'week':
                    posts = list(subreddit.top(time_filter='week', limit=query_limit))
                elif time_period == 'month':
                    posts = list(subreddit.top(time_filter='month', limit=query_limit))
                elif time_period == 'year':
                    posts = list(subreddit.top(time_filter='year', limit=query_limit))
                elif time_period == 'all':
                    posts = list(subreddit.top(time_filter='all', limit=query_limit))
                else:
                    # Default to hot posts if time period isn't recognized
                    posts = list(subreddit.hot(limit=query_limit))

                # If we got here without an exception, break the retry loop
                break

            except RequestException as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait longer between each retry
                    sleep_time = 5 * (2 ** attempt)
                    print(f"Request failed, retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    # We've used all our retries
                    print(f"Failed after {max_retries} attempts. Trying alternative method...")
                    try:
                        # Try using 'hot' as a fallback
                        posts = list(subreddit.hot(limit=query_limit))
                    except Exception as e2:
                        print(f"Alternative method also failed: {str(e2)}")
                        return []  # Return empty list if all attempts fail

        return posts

    def _is_relevant(self, post):
        """Check if a post is relevant to mental health concerns based on content"""
        try:
            # Combine title and selftext for checking
            content = (post.title + " " + post.selftext).lower()

            # Check if any keyword is in the content
            for keyword in MENTAL_HEALTH_KEYWORDS:
                if keyword.lower() in content:
                    return True

            return False
        except Exception as e:
            # In case of attribute error or other issue, be conservative and include the post
            print(f"Error checking relevance: {str(e)}")
            return True

    def _process_post(self, post):
        """Extract relevant data from a Reddit post"""
        try:
            return {
                'post_id': post.id,
                'timestamp': datetime.datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'title': post.title,
                'content': post.selftext,
                'likes': post.score,
                'comments': post.num_comments,
                'subreddit': post.subreddit.display_name,
                'author': str(post.author) if post.author else "[deleted]",
                'permalink': post.permalink,
                'url': post.url
            }
        except Exception as e:
            print(f"Error processing post: {str(e)}")
            # Return a minimal post with available data
            return {
                'post_id': getattr(post, 'id', 'unknown'),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'title': getattr(post, 'title', 'unknown'),
                'content': getattr(post, 'selftext', ''),
                'likes': getattr(post, 'score', 0),
                'comments': getattr(post, 'num_comments', 0),
                'subreddit': getattr(post, 'subreddit', 'unknown'),
                'author': 'unknown',
                'permalink': getattr(post, 'permalink', ''),
                'url': getattr(post, 'url', '')
            }

    def clean_data(self):
        """Clean and preprocess the collected data"""
        print("Cleaning and preprocessing data...")

        if not self.data:
            print("No data to clean.")
            return self.data

        try:
            stop_words = set(stopwords.words('english'))

            for post in self.data:
                # Get the content (combine title and content)
                content = post.get('title', '') + ' ' + post.get('content', '')

                # Clean the content
                cleaned_content = self._clean_text(content)

                # Remove stopwords
                tokens = word_tokenize(cleaned_content)
                filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
                cleaned_content = ' '.join(filtered_tokens)

                # Update the post with cleaned content
                post['cleaned_content'] = cleaned_content

            print("Data cleaning completed")
        except Exception as e:
            print(f"Error during data cleaning: {str(e)}")
            # Add empty cleaned_content field if cleaning fails
            for post in self.data:
                if 'cleaned_content' not in post:
                    post['cleaned_content'] = ''

        return self.data

    def _clean_text(self, text):
        """Clean text by removing emojis, special characters, URLs, etc."""
        try:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)

            # Remove emojis
            text = text.encode('ascii', 'ignore').decode('ascii')

            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)

            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return text  # Return original text if cleaning fails

    def save_data(self, filename=None):
        """Save the data to a file in the specified format"""
        if not self.data:
            print("No data to save.")
            return None

        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reddit_mental_health_data_{timestamp}"

        try:
            if self.output_format == 'csv':
                self._save_as_csv(filename)
            elif self.output_format == 'json':
                self._save_as_json(filename)
            else:
                raise ValueError(f"Output format {self.output_format} not supported. Choose 'csv' or 'json'")

            return filename
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return None

    def _save_as_csv(self, filename):
        """Save data as CSV"""
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create full path in T1_output directory
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                # Define CSV headers based on data structure
                fieldnames = list(self.data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data)

            print(f"Data saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")
            # Try with a different encoding if utf-8 fails
            try:
                with open(filepath, 'w', newline='', encoding='latin-1') as f:
                    fieldnames = list(self.data[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.data)
                print(f"Data saved to {filepath} using latin-1 encoding")
            except Exception as e2:
                print(f"Error saving CSV with alternative encoding: {str(e2)}")

    def _save_as_json(self, filename):
        """Save data as JSON"""
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Create full path in T1_output directory
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)

            print(f"Data saved to {filepath}")
        except Exception as e:
            print(f"Error saving JSON: {str(e)}")
            # Try with a different encoding if utf-8 fails
            try:
                with open(filepath, 'w', encoding='latin-1') as f:
                    json.dump(self.data, f, indent=4)
                print(f"Data saved to {filepath} using latin-1 encoding")
            except Exception as e2:
                print(f"Error saving JSON with alternative encoding: {str(e2)}")

    def _save_subreddit_stats(self):
        """Save subreddit statistics to CSV"""
        stats_file = os.path.join(output_dir, 'subreddit_statistics.csv')
        
        # Prepare statistics for CSV
        stats_data = []
        for subreddit, stats in self.subreddit_stats.items():
            row = {
                'subreddit': subreddit,
                'total_posts_found': stats['total_posts_found'],
                'relevant_posts': stats['relevant_posts'],
                'total_likes': stats['total_likes'],
                'total_comments': stats['total_comments'],
                'engagement_rate': (stats['total_likes'] + stats['total_comments']) / stats['total_posts_found'] if stats['total_posts_found'] > 0 else 0,
                'relevance_rate': (stats['relevant_posts'] / stats['total_posts_found'] * 100) if stats['total_posts_found'] > 0 else 0,
                'top_keywords': ', '.join([f"{k}({v})" for k, v in stats['keywords_found'].most_common(5)])
            }
            stats_data.append(row)

        # Create DataFrame and save to CSV
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_file, index=False)
        print(f"\nSubreddit statistics saved to {stats_file}")
        
        # Print summary
        print("\nSubreddit Statistics Summary:")
        print("-----------------------------")
        for row in stats_data:
            print(f"\nr/{row['subreddit']}:")
            print(f"  Total Posts: {row['total_posts_found']}")
            print(f"  Relevant Posts: {row['relevant_posts']}")
            print(f"  Engagement Rate: {row['engagement_rate']:.2f}")
            print(f"  Relevance Rate: {row['relevance_rate']:.1f}%")
            print(f"  Top Keywords: {row['top_keywords']}")


def main():
    parser = argparse.ArgumentParser(description='Extract mental health related posts from Reddit')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'],
                        help='Output file format (default: csv)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of posts to extract per subreddit (default: 100)')
    parser.add_argument('--time', type=str, default='week',
                        choices=['hour', 'day', 'week', 'month', 'year', 'all'],
                        help='Time period to search within (default: week)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated based on timestamp)')
    parser.add_argument('--subreddits', type=str, nargs='+',
                        help='List of subreddits to search (space-separated)')
    parser.add_argument('--retries', type=int, default=3,
                        help='Maximum number of retry attempts for API requests (default: 3)')

    args = parser.parse_args()

    try:
        # Create the extractor
        extractor = RedditExtractor(output_format=args.format)

        # Extract the data
        extractor.extract_data(
            query_limit=args.limit,
            time_period=args.time,
            subreddits=args.subreddits,
            max_retries=args.retries
        )

        # Clean the data
        extractor.clean_data()

        # Save the data
        extractor.save_data(filename=args.output)

        print("Process completed successfully!")
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        print("Please check your Reddit API credentials and network connection.")


if __name__ == "__main__":
    main()