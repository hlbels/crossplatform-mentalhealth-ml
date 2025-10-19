# Cross Platform Robustness and Fairness of Mental Health Text Classifiers

## 📋 Project Overview
This project investigates cross-platform generalization of machine learning models for mental health detection from social media text. We train models on Reddit mental health posts and test them on Twitter data to identify and mitigate platform-specific biases.
---

### Reddit Dataset

**Source:**  
[Reddit Mental Health Dataset – Zenodo](https://zenodo.org/records/3941387)

This dataset contains posts from multiple mental health–related subreddits, along with one control subreddit (r/jokes).  
Each file includes TF–IDF features (256 dimensions).

**Included Files**

| File Name | Description |
|------------|-------------|
| `addiction_2019_features_tfidf_256` | Posts from r/addiction (TF–IDF 256 features). |
| `adhd_2019_features_tfidf_256` | Posts from r/ADHD (TF–IDF 256 features). |
| `alcoholism_2019_features_tfidf_256` | Posts from r/alcoholism (TF–IDF 256 features). |
| `anxiety_2019_features_tfidf_256` | Posts from r/anxiety (TF–IDF 256 features). |
| `jokes_2019_features_tfidf_256` | Posts from r/jokes, used as the control dataset. |
| `mentalhealth_2019_features_tfidf_256` | Mixed posts from multiple mental health subreddits. |
| `conspiracy_2019_features_tfidf_256` | Posts from r/conspiracy, used as the control dataset. |
| `fitness_2019_features_tfidf_256` | Posts from r/fitness, used as the control dataset. |
| `meditation_2019_features_tfidf_256` | Posts from r/meditation, used as the control dataset. |
| `relationships_2019_features_tfidf_256` | Posts from r/relationship, used as the control dataset. |
| `teaching_2019_features_tfidf_256` | Posts from r/teaching, used as the control dataset. |
| `suicidewatch_2019_features_tfidf_256` | Posts from r/suicidewatch, used as the control dataset. |


**Notebook:**  
`reddit_eda.ipynb` performs:
- Loading and inspection of subreddit files  
- Cleaning and normalization of text data  
- Exploratory Data Analysis (EDA): class balance, post lengths, frequent terms, and TF–IDF statistics  
- Visualization of key results  

Outputs include figures and statistics to be used in the Overleaf report section.

---

### Twitter Dataset 
**Source:**  
https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media

This Kaggle - Mental Health Social Media Dataset, is in uncleaned format and is collected using Twitter API. The Tweets has been filtered to keep only the English context. It targets mental health classification of the user at Tweet-level. Dataset size is 20,000 annotated English tweets (10,000 mental health-related, 10,000 control) and the time Period is  August 2015

**Notebook:**  
`twitter_eda.ipynb` performs:
- Tweet-level analysis of length, complexity, and linguistic patterns
- Social features analysis (followers, retweets, favorites)
- TF-IDF analysis for mental health vs. control tweets
- Readability metrics comparison with Reddit dataset
- Cross-platform comparison visualizations

**1. Basic Statistics**
- Character length analysis (avg: 61.30 for control, 64.14 for mental health)
- Word count statistics (avg: 11.10 words for control, 11.73 for mental health)
- Comparison with Reddit posts (~20x shorter than Reddit's 250-word average)

**2. Social Features Analysis**
- User follower counts (average: 900 followers)
- Friend counts (average: 782 friends)
- Retweet patterns (average: 1,437.93 retweets)
- Favorite counts and status counts
- Distribution analysis by label (mental health vs. control)

**3. TF-IDF Analysis**
- **Mental Health Tweets (Label 1):** "depression" (0.0267), "like" (0.0170), "just" (0.0166), "love" (0.0139), "treatments" (0.0110)
- **Control Tweets (Label 0):** "user" (0.0491), "like" (0.0151), "just" (0.0144), "trump" (0.0133), "thank" (0.0121)
- Clear vocabulary separation demonstrating classification utility

**4. Distribution Analysis**
- Tweet length distribution by label
- Character and word count histograms
- Comparison of mental health vs. control tweet characteristics

**5. Readability Analysis**
- Flesch-Kincaid Grade Level (Grade 7-8, compared to Reddit's Grade 10-11)
- Flesch Reading Ease (55-65, higher than Reddit's 45-55)
- Automated Readability Index (ARI)
- Coleman-Liau Index
- Correlation analysis between readability metrics (r > 0.85 between ARI, Coleman-Liau, and FK Grade)

**6. Cross-Platform Comparison**
- Length: Twitter (11-12 words) vs. Reddit (250 words)
- Readability: Twitter (Grade 7-8) vs. Reddit (Grade 10-11)
- Context richness: Twitter's character limits promote concise communication
- Social features: Twitter provides rich metadata not available in Reddit dataset
