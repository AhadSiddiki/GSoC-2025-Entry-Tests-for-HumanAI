import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import argparse
from collections import Counter

# Ensure NLTK resources are downloaded
nltk.download('vader_lexicon', quiet=True)

# High-risk crisis terms and phrases
HIGH_RISK_TERMS = [
    "kill myself", "end my life", "suicide", "suicidal", "don't want to live",
    "don't want to be alive", "want to die", "better off dead", "end it all",
    "no reason to live", "can't take it anymore", "going to end it",
    "plan to kill", "hurt myself", "self harm", "cutting myself", "overdose"
]

# Moderate concern terms
MODERATE_CONCERN_TERMS = [
    "need help", "struggling", "lost", "hopeless", "worthless", "depressed",
    "depression", "anxiety", "anxious", "panic attack", "overwhelmed",
    "therapy", "medication", "therapist", "psychiatrist", "counseling",
    "mental health", "worried", "stress", "desperate"
]


class SentimentRiskClassifier:
    def __init__(self, input_file):
        """Initialize the classifier with input file"""
        self.input_file = input_file
        self.data = None
        self.sia = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

    def load_data(self):
        """Load data from CSV file"""
        # Check file extension to determine format
        if self.input_file.endswith('.csv'):
            self.data = pd.read_csv(self.input_file)
        elif self.input_file.endswith('.json'):
            self.data = pd.read_json(self.input_file)
        else:
            raise ValueError("Input file must be CSV or JSON")

        print(f"Loaded {len(self.data)} posts from {self.input_file}")
        return self.data

    def analyze_sentiment(self):
        """Apply VADER sentiment analysis to the posts"""
        print("Analyzing sentiment...")

        # Combine title and content for analysis
        self.data['full_text'] = self.data['title'] + ' ' + self.data['content'].fillna('')

        # Apply VADER sentiment analysis
        self.data['sentiment_scores'] = self.data['full_text'].apply(lambda x: self.sia.polarity_scores(x))

        # Extract compound score
        self.data['compound_score'] = self.data['sentiment_scores'].apply(lambda x: x['compound'])

        # Classify sentiment based on compound score
        self.data['sentiment'] = self.data['compound_score'].apply(self._classify_sentiment)

        print("Sentiment analysis completed")
        return self.data

    @staticmethod
    def _classify_sentiment(compound_score):
        """Classify sentiment based on compound score"""
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def classify_risk(self):
        """Classify posts by risk level using term matching and TF-IDF"""
        print("Classifying risk levels...")

        # Simple term matching for high-risk and moderate concern
        self.data['high_risk_matches'] = self.data['full_text'].apply(
            lambda x: self._count_term_matches(x.lower(), HIGH_RISK_TERMS))

        self.data['moderate_concern_matches'] = self.data['full_text'].apply(
            lambda x: self._count_term_matches(x.lower(), MODERATE_CONCERN_TERMS))

        # Apply TF-IDF to get important terms in each post
        tfidf_matrix = self.tfidf.fit_transform(self.data['full_text'])
        feature_names = self.tfidf.get_feature_names_out()

        # Get top terms for each document
        self.data['top_terms'] = self._get_top_terms(tfidf_matrix, feature_names)

        # Classify risk level based on term matches and sentiment
        self.data['risk_level'] = self.data.apply(self._determine_risk_level, axis=1)

        print("Risk classification completed")
        return self.data

    def _get_top_terms(self, tfidf_matrix, feature_names, top_n=10):
        """Get top TF-IDF terms for each document"""
        top_terms_list = []

        for i in range(tfidf_matrix.shape[0]):
            doc_vector = tfidf_matrix[i].toarray()[0]
            # Get indices of top terms
            top_indices = doc_vector.argsort()[-top_n:][::-1]
            # Get the actual terms
            top_terms = [feature_names[idx] for idx in top_indices]
            top_terms_list.append(top_terms)

        return top_terms_list

    @staticmethod
    def _count_term_matches(text, term_list):
        """Count how many terms from the term list appear in the text"""
        count = 0
        for term in term_list:
            if term in text:
                count += 1
        return count

    @staticmethod
    def _determine_risk_level(row):
        """Determine risk level based on term matches and sentiment"""
        # High risk if any high-risk terms are found
        if row['high_risk_matches'] > 0:
            return "High-Risk"

        # Moderate concern if moderate terms are found or negative sentiment
        if row['moderate_concern_matches'] > 0 or row['sentiment'] == "Negative":
            return "Moderate Concern"

        # Otherwise low concern
        return "Low Concern"

    def generate_statistics(self):
        """Generate statistics on sentiment and risk classification"""
        # Count by sentiment
        sentiment_counts = self.data['sentiment'].value_counts()

        # Count by risk level
        risk_counts = self.data['risk_level'].value_counts()

        # Cross-tabulation of sentiment and risk
        sentiment_risk_crosstab = pd.crosstab(self.data['sentiment'], self.data['risk_level'])

        stats = {
            'sentiment_counts': sentiment_counts,
            'risk_counts': risk_counts,
            'sentiment_risk_crosstab': sentiment_risk_crosstab
        }

        return stats

    def plot_distributions(self, output_dir='T2_output', show_plots=True):
        """Create plots for sentiment and risk distributions"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create figure for subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot sentiment distribution (fixed for seaborn deprecation warning)
        sns.countplot(x='sentiment', data=self.data, ax=axes[0, 0], hue='sentiment', palette='viridis', legend=False)
        axes[0, 0].set_title('Distribution of Sentiment')
        axes[0, 0].set_ylabel('Count')

        # Plot risk level distribution (fixed for seaborn deprecation warning)
        sns.countplot(x='risk_level', data=self.data, ax=axes[0, 1], hue='risk_level', palette='rocket', legend=False)
        axes[0, 1].set_title('Distribution of Risk Levels')
        axes[0, 1].set_ylabel('Count')

        # Plot sentiment by risk level
        sentiment_risk_crosstab = pd.crosstab(self.data['sentiment'], self.data['risk_level'])
        sentiment_risk_crosstab.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='viridis')
        axes[1, 0].set_title('Sentiment by Risk Level')
        axes[1, 0].set_ylabel('Count')

        # Plot risk level by subreddit
        subreddit_risk_crosstab = pd.crosstab(self.data['subreddit'], self.data['risk_level'])
        subreddit_risk_crosstab.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='rocket')
        axes[1, 1].set_title('Risk Level by Subreddit')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save the figure as PNG with high DPI for quality
        plot_path = os.path.join(output_dir, 'sentiment_risk_distribution.png')
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")

        # Display the plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # Create a separate figure for a heatmap visualization of sentiment vs. risk
        plt.figure(figsize=(10, 8))

        # Create a pivot table for the heatmap
        heatmap_data = pd.crosstab(self.data['sentiment'], self.data['risk_level'])

        # Plot the heatmap
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt='d')
        plt.title('Sentiment vs. Risk Level Distribution')
        plt.tight_layout()

        # Save the heatmap as PNG with high DPI
        heatmap_path = os.path.join(output_dir, 'sentiment_risk_heatmap.png')
        plt.savefig(heatmap_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {heatmap_path}")

        # Display the heatmap if requested
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Create a pie chart for risk levels
        plt.figure(figsize=(10, 8))
        self.data['risk_level'].value_counts().plot.pie(
            autopct='%1.1f%%',
            shadow=True,
            explode=[0.05, 0.05, 0.05],  # Slight explode for all segments
            colors=sns.color_palette('rocket', 3),
            startangle=90,
            textprops={'fontsize': 14}
        )
        plt.title('Distribution of Risk Levels', fontsize=16)
        plt.ylabel('')  # Remove ylabel

        # Save the pie chart as PNG
        pie_path = os.path.join(output_dir, 'risk_level_pie_chart.png')
        plt.savefig(pie_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to {pie_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Return paths to the created plots
        return {
            'distribution_plot': plot_path,
            'heatmap': heatmap_path,
            'pie_chart': pie_path
        }

    def save_results(self, output_format='csv', output_dir='T2_output'):
        """Save the classification results"""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mental_health_sentiment_risk_{timestamp}"

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save full data with classifications
        output_path = os.path.join(output_dir, filename)

        if output_format.lower() == 'csv':
            # Save as CSV
            # Convert complex columns to strings to avoid serialization issues
            self.data['sentiment_scores'] = self.data['sentiment_scores'].apply(lambda x: str(x))
            self.data['top_terms'] = self.data['top_terms'].apply(lambda x: str(x))

            self.data.to_csv(f"{output_path}.csv", index=False)
            print(f"Results saved to {output_path}.csv")

        elif output_format.lower() == 'json':
            # For JSON, convert sentiment_scores to individual columns
            self.data['pos_score'] = self.data['sentiment_scores'].apply(
                lambda x: eval(x).get('pos', 0) if isinstance(x, str) else x.get('pos', 0))
            self.data['neg_score'] = self.data['sentiment_scores'].apply(
                lambda x: eval(x).get('neg', 0) if isinstance(x, str) else x.get('neg', 0))
            self.data['neu_score'] = self.data['sentiment_scores'].apply(
                lambda x: eval(x).get('neu', 0) if isinstance(x, str) else x.get('neu', 0))

            # Drop the original sentiment_scores column
            self.data = self.data.drop('sentiment_scores', axis=1)

            # Convert top_terms to string if it's not already
            self.data['top_terms'] = self.data['top_terms'].apply(lambda x: str(x))

            self.data.to_json(f"{output_path}.json", orient='records', indent=4)
            print(f"Results saved to {output_path}.json")

        else:
            raise ValueError("Output format must be 'csv' or 'json'")

        # Save statistics and cross-tabulations
        stats = self.generate_statistics()

        # Save sentiment counts
        stats['sentiment_counts'].to_csv(os.path.join(output_dir, f"sentiment_counts_{timestamp}.csv"))

        # Save risk counts
        stats['risk_counts'].to_csv(os.path.join(output_dir, f"risk_counts_{timestamp}.csv"))

        # Save crosstab
        stats['sentiment_risk_crosstab'].to_csv(os.path.join(output_dir, f"sentiment_risk_crosstab_{timestamp}.csv"))

        return output_path


def main():
    parser = argparse.ArgumentParser(description='Classify Reddit mental health posts by sentiment and risk level')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file path (CSV or JSON from Reddit extraction)')
    parser.add_argument('--output-format', type=str, default='csv', choices=['csv', 'json'],
                        help='Output file format (default: csv)')
    parser.add_argument('--output-dir', type=str, default='T2_output',
                        help='Output directory for results and plots')
    parser.add_argument('--no-show-plots', action='store_true',
                        help='Do not display plots (just save them)')

    args = parser.parse_args()

    # Create classifier
    classifier = SentimentRiskClassifier(args.input)

    # Load data
    classifier.load_data()

    # Analyze sentiment
    classifier.analyze_sentiment()

    # Classify risk level
    classifier.classify_risk()

    # Generate plots
    show_plots = not args.no_show_plots
    classifier.plot_distributions(args.output_dir, show_plots=show_plots)

    # Save results
    classifier.save_results(args.output_format, args.output_dir)

    print("Classification completed successfully!")


# For use in Jupyter notebooks or interactive environments
def process_data(input_file, output_format='csv', output_dir='T2_output'):
    """Process the data and generate plots for interactive environments"""
    # Create classifier
    classifier = SentimentRiskClassifier(input_file)

    # Load data
    classifier.load_data()

    # Analyze sentiment
    classifier.analyze_sentiment()

    # Classify risk level
    classifier.classify_risk()

    # Generate plots (display them)
    classifier.plot_distributions(output_dir, show_plots=True)

    # Save results
    classifier.save_results(output_format, output_dir)

    # Return the data for further analysis
    return classifier.data, classifier.generate_statistics()


if __name__ == "__main__":
    main()