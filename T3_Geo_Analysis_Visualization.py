import pandas as pd
import folium
from folium.plugins import HeatMap
import spacy
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from collections import Counter
import matplotlib.pyplot as plt
import os
import argparse

# Load SpaCy NLP model for location extraction
try:
    nlp = spacy.load("en_core_web_lg")
    print("Loaded SpaCy model successfully")
except:
    print("Installing required SpaCy model...")
    import subprocess

    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

# Initialize geocoder with user-agent and rate limiting
geocoder = Nominatim(user_agent="crisis_mapping_app")
geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1)


class CrisisGeolocation:
    def __init__(self, input_file):
        """Initialize with input CSV file from Reddit extractor"""
        self.input_file = input_file
        self.data = None
        self.location_data = {}
        self.coordinates = []
        self.location_counts = Counter()

    def load_data(self):
        """Load data from CSV file"""
        print(f"Loading data from {self.input_file}...")
        try:
            self.data = pd.read_csv(self.input_file)
            print(f"Loaded {len(self.data)} records")

            # Ensure title and content columns exist and are strings
            if 'title' not in self.data.columns:
                print("Warning: 'title' column missing from data")
                self.data['title'] = ""
            else:
                self.data['title'] = self.data['title'].fillna("").astype(str)

            if 'content' not in self.data.columns:
                print("Warning: 'content' column missing from data")
                self.data['content'] = ""
            else:
                self.data['content'] = self.data['content'].fillna("").astype(str)

            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def extract_locations(self):
        """Extract location mentions from post content using NLP"""
        print("Extracting location mentions from posts...")

        # Combine title and content for processing
        self.data['full_text'] = self.data['title'] + " " + self.data['content']

        # Create locations column
        self.data['locations'] = ""

        # Process each post to extract locations
        locations_found = 0

        for idx, row in self.data.iterrows():
            try:
                # Ensure we have valid text to process
                text = row['full_text']
                if not isinstance(text, str) or not text.strip():
                    continue

                # Process text with SpaCy
                doc = nlp(text)

                # Extract GPE (Geo-Political Entity) and LOC (Location) entities
                locations = []
                for ent in doc.ents:
                    if ent.label_ in ["GPE", "LOC"]:
                        locations.append(ent.text)

                # Store locations with the post
                if locations:
                    self.data.at[idx, 'locations'] = ", ".join(locations)
                    locations_found += 1

                    # Update location counter
                    for loc in locations:
                        self.location_counts[loc] += 1

            except Exception as e:
                print(f"Error processing text at index {idx}: {str(e)}")
                continue

        print(f"Found locations in {locations_found} posts")
        return locations_found > 0

    def geocode_locations(self):
        """Convert extracted locations to geographical coordinates"""
        print("Geocoding locations...")

        # Process all unique locations
        unique_locations = set()
        for locs in self.data['locations'].dropna():
            if isinstance(locs, str) and locs.strip():
                unique_locations.update([loc.strip() for loc in locs.split(',')])

        print(f"Found {len(unique_locations)} unique locations to geocode")

        # Geocode each unique location
        for location in unique_locations:
            try:
                # Skip very generic or likely false positive locations
                if location.lower() in ['here', 'there', 'home', 'house', 'room', 'bathroom']:
                    continue

                geo_result = geocode(location)
                if geo_result:
                    self.location_data[location] = {
                        'latitude': geo_result.latitude,
                        'longitude': geo_result.longitude,
                        'address': geo_result.address
                    }
                    print(f"Geocoded: {location}")
            except Exception as e:
                print(f"Error geocoding {location}: {str(e)}")

        print(f"Successfully geocoded {len(self.location_data)} locations")
        return len(self.location_data) > 0

    def prepare_heatmap_data(self):
        """Prepare data for heatmap by mapping post locations to coordinates"""
        print("Preparing heatmap data...")

        # Process each post with location data
        for idx, row in self.data.iterrows():
            if pd.notna(row['locations']) and isinstance(row['locations'], str) and row['locations'].strip():
                locations = [loc.strip() for loc in row['locations'].split(',')]

                # For each location mentioned in the post
                for loc in locations:
                    if loc in self.location_data:
                        # Get coordinates and add to heatmap data with intensity based on post metrics
                        lat = self.location_data[loc]['latitude']
                        lon = self.location_data[loc]['longitude']

                        # Use post engagement as weight (likes + comments)
                        weight = 1  # Base weight

                        # Safely handle likes and comments
                        likes = row.get('likes', 0)
                        comments = row.get('comments', 0)

                        if pd.notna(likes) and isinstance(likes, (int, float)):
                            weight += min(int(likes), 100) / 20  # Scale likes

                        if pd.notna(comments) and isinstance(comments, (int, float)):
                            weight += min(int(comments), 50) / 10  # Scale comments

                        self.coordinates.append([lat, lon, weight])

        print(f"Prepared {len(self.coordinates)} weighted coordinate points for heatmap")
        return len(self.coordinates) > 0

    def create_heatmap(self, output_file="T3_output/crisis_heatmap.html"):
        """Generate a heatmap visualization of crisis-related posts"""
        print("Generating heatmap...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Create base map centered on average coordinates
        if not self.coordinates:
            print("No coordinates available for heatmap")
            return False

        # Calculate center point for the map
        avg_lat = sum(coord[0] for coord in self.coordinates) / len(self.coordinates)
        avg_lon = sum(coord[1] for coord in self.coordinates) / len(self.coordinates)

        # Create the map
        crisis_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)

        # Add heatmap layer
        HeatMap(self.coordinates).add_to(crisis_map)

        # Add markers for top locations
        for loc, count in self.location_counts.most_common(10):
            if loc in self.location_data:
                lat = self.location_data[loc]['latitude']
                lon = self.location_data[loc]['longitude']
                popup_text = f"<strong>{loc}</strong><br>Mentions: {count}"
                folium.Marker(
                    [lat, lon],
                    popup=popup_text,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(crisis_map)

        # Save the map
        crisis_map.save(output_file)
        print(f"Heatmap saved to {output_file}")

        return True

    def plot_top_locations(self, top_n=5, output_file="T3_output/top_crisis_locations.png"):
        """Generate a bar chart of top locations with crisis mentions"""
        print(f"Generating top {top_n} locations chart...")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Get top locations
        top_locations = self.location_counts.most_common(top_n)

        if not top_locations:
            print("No location data available for chart")
            return False

        # Create the bar chart
        plt.figure(figsize=(10, 6))
        locations = [loc[0] for loc in top_locations]
        counts = [loc[1] for loc in top_locations]

        plt.bar(locations, counts, color='darkred')
        plt.title(f'Top {top_n} Locations with Crisis Mentions')
        plt.xlabel('Location')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the chart
        plt.savefig(output_file)
        print(f"Top locations chart saved to {output_file}")

        return True

    def generate_detailed_location_report(self, output_file="T3_output/location_report.csv"):
        """Generate a detailed report of all geocoded locations and their mention counts"""
        print("Generating detailed location report...")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Create a dataframe with location data
        report_data = []
        for location, count in self.location_counts.items():
            if location in self.location_data:
                report_data.append({
                    'location': location,
                    'mentions': count,
                    'latitude': self.location_data[location]['latitude'],
                    'longitude': self.location_data[location]['longitude'],
                    'full_address': self.location_data[location]['address']
                })

        if report_data:
            report_df = pd.DataFrame(report_data)
            report_df.sort_values('mentions', ascending=False, inplace=True)
            report_df.to_csv(output_file, index=False)
            print(f"Location report saved to {output_file}")
            return True
        else:
            print("No location data available for report")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract and map location data from Reddit mental health posts')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file from Reddit data extractor')
    parser.add_argument('--heatmap', type=str, default='T3_output/crisis_heatmap.html',
                        help='Output heatmap HTML file (default: T3_output/crisis_heatmap.html)')
    parser.add_argument('--chart', type=str, default='T3_output/top_crisis_locations.png',
                        help='Output chart image file (default: T3_output/top_crisis_locations.png)')
    parser.add_argument('--report', type=str, default='T3_output/location_report.csv',
                        help='Output detailed location report (default: T3_output/location_report.csv)')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of top locations to display (default: 5)')

    args = parser.parse_args()

    # Create geolocation processor
    processor = CrisisGeolocation(args.input)

    # Process the data
    if processor.load_data():
        processor.extract_locations()
        processor.geocode_locations()
        processor.prepare_heatmap_data()
        processor.create_heatmap(args.heatmap)
        processor.plot_top_locations(top_n=args.top, output_file=args.chart)
        processor.generate_detailed_location_report(args.report)

        print("\nCrisis geolocation and mapping completed successfully!")
        print(f"- Heatmap: {args.heatmap}")
        print(f"- Top locations chart: {args.chart}")
        print(f"- Detailed location report: {args.report}")
    else:
        print("Processing failed. Please check your input file.")


if __name__ == "__main__":
    main()