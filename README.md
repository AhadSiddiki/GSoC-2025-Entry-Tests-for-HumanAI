# Mental Health Crisis Analysis - GSOC 2025 Test Submission

## Project Title: AI-Powered Behavioral Analysis for Suicide Prevention, Substance Use, and Mental Health Crisis Detection with Longitudinal Geospatial Crisis Trend Analysis

## Project Overview
This project implements a comprehensive pipeline for analyzing mental health-related discussions on Reddit, focusing on crisis detection, sentiment analysis, and geographical visualization.

## Test Requirements Fulfillment

### ✅ Task 1: Social Media Data Extraction & Preprocessing
- Implemented Reddit API integration using PRAW
- Applied keyword filtering for mental health discussions
- Created structured data storage system
- Implemented text preprocessing

### ✅ Task 2: Sentiment & Crisis Risk Classification
- Implemented VADER sentiment analysis
- Created risk level classification system
- Generated statistical visualizations
- Applied TF-IDF for term importance

### ✅ Task 3: Crisis Geolocation & Mapping
- Implemented location extraction using SpaCy NLP
- Created interactive heatmaps with Folium
- Generated geographical statistics
- Visualized regional crisis patterns

### Components
1. Reddit Data Collection (T1_RMHDCscript.py)
2. Sentiment & Risk Analysis (T2_NLPtxtProcessing.py)
3. Geographical Analysis & Visualization
(T3_Geo_Analysis_Visualization.py)

## Project Overview
This project consists of three main components that work together to collect, analyze, and visualize mental health-related discussions from Reddit:

1. T1_RMHDCscript.py - Reddit Data Collection
2. T2_NLPtxtProcessing.py - Natural Language Processing & Sentiment Analysis
3. T3_Geo_Analysis_Visualization.py - Geographical Analysis & Visualization

## Setup and Configuration

### 1. Prerequisites
```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Reddit API Configuration
1. Create a Reddit developer account at https://www.reddit.com/prefs/apps
2. Create a new application (script type)
3. Create a `.env` file in your project root with:
```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT='Mental Health Data Extraction Script v1.0'
```
### Task 1: Social Media Data Extraction & Preprocessing
**Objective**: Extract and process mental health-related discussions from Reddit.

#### Implementation Details
1. **Reddit API Integration**
   - Used PRAW (Python Reddit API Wrapper)
   - Implemented rate limiting and error handling
   - Configured with secure environment variables

2. **Keyword Filtering System**
   ```python
   MENTAL_HEALTH_KEYWORDS = [
       "depressed", "depression", "anxiety", "anxious",
       "overwhelmed", "suicidal", "suicide", "self-harm",
       "addiction", "substance abuse", "mental health",
       "therapy", "panic attack", "bipolar", "ptsd"
   ]
   ```

3. **Data Preprocessing Pipeline**
   - Text cleaning
   - Emoji removal
   - Special character handling
   - Stopword removal
   
   ![Preprocessing Steps](docs/images/preprocessing_pipeline.png)

4. **Output Structure**
   ```json
   {
       "post_id": "abc123",
       "timestamp": "2024-02-20 10:30:00",
       "title": "Need help coping",
       "content": "Processed content text...",
       "subreddit": "mentalhealth",
       "engagement": {
           "likes": 10,
           "comments": 5
       }
   }
   ```
#### Usage
```bash
python T1_RMHDCscript.py --format csv --limit 100 --time week
```
#### Output:
- Saves processed data in `T1_output/reddit_mental_health_data_TIMESTAMP.csv`

![Data Collection Output](docs/images/data_collection_output.png)

### Task 2: Sentiment & Crisis Risk Classification
**Objective**: Analyze post content for sentiment and crisis risk levels.

#### Implementation Details
1. **Sentiment Analysis System**
   - VADER Sentiment Analysis
   - Compound score calculation
   - Sentiment classification thresholds

   ![Sentiment Analysis Process](docs/images/sentiment_analysis.png)

2. **Risk Classification Categories**
   ```python
   HIGH_RISK_TERMS = [
       "kill myself", "end my life", "suicide", "suicidal",
       "don't want to live", "want to die", "better off dead",
       "end it all", "no reason to live", "can't take it anymore"
   ]

   MODERATE_CONCERN_TERMS = [
       "need help", "struggling", "lost", "hopeless",
       "worthless", "depressed", "depression", "anxiety",
       "panic attack", "overwhelmed", "therapy"
   ]
   ```

3. **TF-IDF Implementation**
   - Term frequency analysis
   - Inverse document frequency calculation
   - Feature importance scoring

   ![TF-IDF Analysis](docs/images/tfidf_analysis.png)

4. **Visualization Outputs**
   - Risk level distribution
   - Sentiment analysis results
   - Cross-tabulation statistics

#### Usage
```bash
python T2_NLPtxtProcessing.py --input input_file.csv
```
#### Output:
- Sentiment analysis results
- Risk classification
- Statistical visualizations in `T2_output/`

#### Sample Visualizations
![Sentiment Distribution](docs/images/sentiment_distribution.png)
![Risk Level Analysis](docs/images/risk_level_analysis.png)

### Task 3: Crisis Geolocation & Mapping
**Objective**: Create geographical visualizations of crisis-related discussions.

#### Implementation Details
1. **Location Extraction**
   - SpaCy NLP for entity recognition
   - Location validation
   - Coordinate mapping

   ![Location Extraction Process](docs/images/location_extraction.png)

2. **Geocoding System**
   ```python
   geocoder = Nominatim(user_agent="crisis_mapping_app")
   geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1)
   ```

3. **Heatmap Generation**
   - Coordinate plotting
   - Intensity calculation
   - Interactive markers

   ![Heatmap Generation Process](docs/images/heatmap_process.png)

4. **Statistical Analysis**
   - Regional distribution
   - Crisis hotspots
   - Temporal patterns

#### Usage
```bash
python T3_Geo_Analysis_Visualization.py --input analyzed_data.csv
```
#### Output:
- Interactive heatmap (`T3_output/crisis_heatmap.html`)
- Location statistics (`T3_output/location_report.csv`)
- Visualization charts (`T3_output/top_crisis_locations.png`)

#### Sample Maps and Charts
![Crisis Heatmap](docs/images/crisis_heatmap.png)
![Top Locations Chart](docs/images/top_locations.png)

## Data Flow
[Reddit API] → T1 (Data Collection)
    → T1_output/data.csv
    → T2 (Sentiment Analysis)
        → T2_output/analyzed_data.csv
        → T3 (Geo Analysis)
            → T3_output/visualizations

## Technical Implementation


### API Integration Details
1. **Reddit API Configuration**
   ```python
   self.reddit = praw.Reddit(
       client_id=os.getenv('REDDIT_CLIENT_ID'),
       client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
       user_agent=os.getenv('REDDIT_USER_AGENT'),
       timeout=30
   )
   ```

2. **Geocoding Implementation**
   ```python
   geocoder = Nominatim(user_agent="crisis_mapping_app")
   geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1)
   ```

## Error Handling

### API Requests
- Rate limiting compliance
- Exponential backoff
- Maximum retry attempts

### Data Processing
- Input validation
- Missing data handling
- Error logging

## Output Directory Structure
project_root/\
├── docs/\
│ ├── images/\
│ │ ├── T1_output.png\
│ │ ├── T2_output.png\
│ │ └── ...\
│ └── Readme.md\
├── output/\
│ ├── T1_output/\
│ │ └── reddit_mental_health_data_.csv\
│ ├── T2_output/\
│ │ ├── sentiment_analysis_.csv\
│ │ └── visualizations/\
│ └── T3_output/\
│ ├── crisis_heatmap.html\
│ └── location_report.csv\
└── scripts/\
├── T1_RMHDCscript.py\
├── T2_NLPtxtProcessing.py\
└── T3_Geo_Analysis_Visualization.py\

## Sample Results

### 1. Data Collection Statistics
![Collection Stats](docs/images/collection_stats.png)
- Structured CSV/JSON files containing:
  - Post ID
  - Timestamp
  - Content
  - Engagement metrics
  - Preprocessed text


### 2. Sentiment Analysis Results
![Sentiment Results](docs/images/sentiment_results.png)
- Risk level distribution
- Sentiment analysis results
- Cross-tabulation statistics
- Visualization plots

### 3. Geographical Distribution
![Geo Distribution](docs/images/geo_distribution.png)
- Interactive heatmap
- Top locations chart
- Detailed location reports
- Regional statistics

## Data Privacy Considerations

### Sensitive Information
- Personal identifiers removed
- Location data aggregated
- Author names anonymized

### Data Storage
- Local storage only
- Temporary file cleanup
- Secure credential handling

## Running the Complete Pipeline

1. First, collect data:
```bash
python T1_RMHDCscript.py --format csv --limit 100
```
   ![T1 Output](docs/images/t1_output.png)

2. Then, analyze sentiment:
```bash
python T2_NLPtxtProcessing.py --input T1_output/reddit_mental_health_data_*.csv
```
   ![T2 Output](docs/images/t2_output.png)

3. Finally, generate geographical analysis:
```bash
python T3_Geo_Analysis_Visualization.py --input T2_output/mental_health_sentiment_*.csv
```
   ![T3 Output](docs/images/t3_output.png)

## Error Handling
### API Requests
- Rate limiting compliance
- Exponential backoff
- Maximum retry attempts

### Data Processing
- Input validation
- Missing data handling
- Error logging

## Requirements
```txt
# API & Data Processing
praw>=7.0.0
pandas>=1.0.0
numpy>=1.18.0

# NLP & Analysis
nltk>=3.6.0
spacy>=3.0.0
scikit-learn>=0.24.0

# Visualization
matplotlib>=3.3.0
seaborn>=0.11.0
folium>=0.12.0

# Geolocation
geopy>=2.2.0

# Utilities
python-dotenv>=0.19.0
```

## Future Improvements
1. Analysis Enhancement
   - Advanced NLP techniques
   - Machine learning integration
   - Real-time processing

2. Visualization Updates
   - Interactive dashboards
   - Custom plotting options
   - Additional chart types

3. Feature Additions
   - Topic modeling
   - User interaction analysis
   - Temporal pattern detection


## Contact Information
For questions about this GSOC test submission, please contact:
- GitHub: [https://github.com/AhadSiddiki]
- Email: [ahad.siddiki25@gmail.com]
- LinkedIn: [http://www.linkedin.com/in/ahad-siddiki/]
- Instagram: [www.instagram.com/ahad.siddiki/]
