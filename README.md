# crossplatform-mentalhealth-ml

## Data Collection and Preprocessing

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

**Notebook:**  
`reddit_eda.ipynb` performs:
- Loading and inspection of subreddit files  
- Cleaning and normalization of text data  
- Exploratory Data Analysis (EDA): class balance, post lengths, frequent terms, and TF–IDF statistics  
- Visualization of key results  

Outputs include figures and statistics to be used in the Overleaf report section.

---

### Twitter Dataset (to be added)

The Twitter dataset and its corresponding preprocessing notebook will be added here once completed by the assigned team member.
