<div align="center">

# Data Science Projects

### Eight end-to-end ML projects — led by ride-demand forecasting & churn retention — spanning time series, churn modeling, GenAI/RAG, NLP, deep learning, regression, and recommendation systems

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-GenAI_RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Search-FF5555?style=for-the-badge)

<br/>

[![Profile](https://img.shields.io/badge/GitLab-Profile-FC6D26?style=for-the-badge&logo=gitlab&logoColor=white)](https://gitlab.com/ansht_121)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Site-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://ansh-thakur-netlify.netlify.app/)
[![Data Analytics](https://img.shields.io/badge/GitLab-Data_Analytics-6C8CFF?style=for-the-badge&logo=gitlab&logoColor=white)](https://gitlab.com/ansht_121/data-analytics)

</div>

---

## Overview

### Cross-Listed Analytics & Predictive Projects

| # | Project | Problem Type | Approach | Result |
|---|---------|--------------|----------|--------|
| 1 | [Ride Demand and Revenue Forecasting](#1--ride-demand-and-revenue-forecasting) | Time series · Clustering · Regression | SARIMA · K-Means · Random Forest · LLM Forecast Guard | **R² = 0.972** fare model · **MAPE 8.3%** forecast · **GenAI anomaly briefs** |
| 2 | [Subscription Churn Prediction](#2--subscription-churn-prediction--retention-analytics) | Binary classification · RAG | LogReg + Random Forest · ChromaDB RAG + LLM offers | **ROC-AUC 0.839** · 80.0% accuracy · **GenAI retention gateway** |

> **Note:** These two projects are cross-listed from the [Data Analytics repository](https://gitlab.com/ansht_121/data-analytics) â€” the SQL / EDA / dashboard layers live there; this repository contains the full predictive modeling layer.

### Standalone Machine Learning Projects

| # | Project | Problem Type | Approach | Result |
|---|---------|--------------|----------|--------|
| 3 | [Image Classification](#3--image-classification--handwritten-digit-recognition) | Multi-class classification | CNN (TensorFlow/Keras) | **98.83%** test accuracy |
| 4 | [Spam Email Detection](#4--spam-email-detection) | Binary text classification | TF-IDF + Logistic Regression | **96.32%** test accuracy |
| 5 | [Sales Prediction](#5--sales-prediction) | Regression | Linear Regression | **RÂ² = 0.91** Â· MSE 2.86 |
| 6 | [Car Price Prediction](#6--car-price-prediction) | Regression | Random Forest Regressor | MAE Â· MSE Â· RÂ² evaluated |
| 7 | [Sentiment Analysis](#7--sentiment-analysis--tweet-classification) | NLP / text classification | TF-IDF + Logistic Regression | **71%** accuracy, 0.70 weighted F1 |
| 8 | [Music Recommendation](#8--music-recommendation-system) | Recommendation | Content-based, cosine similarity | Top-K similar songs |

---

## 1 Â· Ride Demand and Revenue Forecasting

**Problem:** A ride-hailing operator needs to know *when* to surge-price, *where* to position premium vehicles, and *which* fleet tier to grow â€” 35,000 trip records, five pickup zones, one simulated year.

*Cross-listed from the [Data Analytics repo](https://gitlab.com/ansht_121/data-analytics) â€” the SQL / Power BI dashboard layer lives there; this folder contains the full predictive modeling layer.*

![Ride Demand Dashboard](./01_ride-demand-and-revenue-forecasting/image/PowerBi_Dashboard.png)
*Executive Ride Demand & Dynamic Fare Dashboard â€” Hourly trip surge, vehicle tier revenue efficiency, and airport route analysis.*

### Pipeline

```
35,000 trips (Janâ€“Dec)
  â†“ SQL cleaning pipeline (ROW_NUMBER() dedup safeguard, type coercion)
  â†“ SARIMA(1,1,1)(1,0,1,7) â€” daily demand forecast, weekly seasonality
  â†“ K-Means over zoneÃ—hour cells â€” demand segmentation (silhouette-selected k)
  â†“ RandomForestRegressor â€” route-level fare prediction
  â†“ All validated on a held-out Novâ€“Dec window (no leakage)
```

### Results (held-out)

| Model | Performance |
|-------|-------------|
| **SARIMA demand forecast** | MAE 7.6 Â· RMSE 9.6 Â· **MAPE 8.3%** â€” ~20% error reduction vs the seasonal-naive baseline (MAE 9.5) |
| **K-Means demand clusters** | k=2 Â· **silhouette 0.675** â€” premium long-haul cells (avg â‚¹507 / 24.9 km) vs high-volume short-trip cells (avg â‚¹331 / 15.6 km) |
| **RF fare prediction** | **RÂ² = 0.972** Â· RMSE â‚¹37.4 â€” prediction error ~13% of a typical fare |

> **Honest read:** SARIMA residuals pass Ljung-Box (p = 0.54 / 0.86) â€” no leftover autocorrelation. The fare model's high RÂ² reflects that distance and duration genuinely determine fare; the value is the *expected vs charged* fare gap as an operational lever, not the score itself.

**How a stakeholder uses this:** ops teams re-time driver incentives to 4:30 PM (before the growth window opens) and position premium fleet for 5â€“7 PM; the fare model flags routes where charged fares drift below expectation.

ðŸ“ [`01_ride-demand-and-revenue-forecasting/`](./01_ride-demand-and-revenue-forecasting/)

---

## 2 Â· Subscription Churn Prediction & Retention Analytics

**Problem:** A subscription business (real-world telco data, 7,043 customers) is losing ~26% of customers. Predict who is at risk, why, and what revenue is on the table. **The deliverable is a revenue-at-risk ranking that tells the retention team where spend saves the most dollars â€” month-to-month + fiber-optic customers alone are 72% of the exposure.**

*Cross-listed from the [Data Analytics repo](https://gitlab.com/ansht_121/data-analytics) â€” the SQL / EDA / Excel dashboard layer lives there; this folder contains the full ML pipeline.*

![Subscription Churn Dashboard](./02_subscription-retention-prediction/image/Churn_Dashboard_screenshot.png)
*Executive Customer Churn & Retention Dashboard â€” KPI cards, risk segmentation by contract/service, and revenue-at-risk analysis.*

### Pipeline

```
Telco Customer Churn (7,043 Ã— 21)
  â†“ Cleaning & feature engineering (tenure bands, service counts, charge ratios)
  â†“ Churn_EDA_Analysis.ipynb â€” KPI baseline, segment analysis, statistical validation
  â†“ Churn_Prediction_Analysis.ipynb â€” Logistic Regression + Random Forest
  â†“ Modular src/ pipeline: data_processing â†’ feature_engineering â†’ modeling
    â†’ evaluation â†’ explainability â†’ risk_scoring
  â†“ pytest suite covering the pipeline modules
```

### Results (held-out)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Logistic Regression** | **0.800** | 0.640 | **0.561** | 0.598 | **0.839** |
| Random Forest | 0.797 | **0.652** | 0.511 | 0.573 | 0.837 |

> **Honest read:** recall is the metric that matters here â€” missing a churner costs the full customer value, while a false alarm costs one retention call. Logistic Regression wins on ROC-AUC (0.839) and recall (0.561) despite Random Forest's slight precision edge. Risk scores feed a revenue-at-risk segmentation (`churn_segment_revenue.csv`) rather than stopping at a leaderboard number.

**How a stakeholder uses this:** the retention team gets a ranked call list with revenue exposure per customer â€” spend prioritised by dollars at risk, not by model score.

ðŸ“ [`02_subscription-retention-prediction/`](./02_subscription-retention-prediction/)

---

## 3 Â· Image Classification â€” Handwritten Digit Recognition

**Problem:** Automatically recognize handwritten digits (0â€“9) from images â€” the foundational task behind postal code sorting and cheque digitization.

**Dataset:** MNIST â€” 60,000 training + 10,000 test grayscale images (28Ã—28 px), loaded directly via `keras.datasets`.

### Model Architecture

```
Input (28Ã—28Ã—1)
  â†“ Conv2D(32, 3Ã—3, ReLU)  â†’  MaxPooling2D(2Ã—2)
  â†“ Conv2D(64, 3Ã—3, ReLU)  â†’  MaxPooling2D(2Ã—2)
  â†“ Conv2D(64, 3Ã—3, ReLU)
  â†“ Flatten  â†’  Dense(64, ReLU)  â†’  Dense(10, Softmax)
```

**Optimizer:** Adam Â· **Loss:** Sparse categorical cross-entropy Â· **Epochs:** 5

### Results

| Epoch | Train Accuracy | Train Loss |
|-------|----------------|------------|
| 1 | 88.98% | 0.3449 |
| 2 | 98.46% | 0.0504 |
| 3 | 99.01% | 0.0332 |
| 4 | 99.21% | 0.0253 |
| 5 | 99.43% | 0.0185 |

> **Test accuracy: 98.83%** (loss 0.0491) â€” the small train/test gap indicates the model generalized well without significant overfitting.

**Key steps:** pixel normalization to `[0,1]` Â· channel-dimension reshaping Â· prediction visualization with Matplotlib Â· model exported to `mnist_model.h5`

**How a stakeholder uses this:** a proof-of-work CNN for OCR intake (form digitisation, cheque processing) â€” the same architecture transfers to document classification pipelines.

ðŸ““ [`03_image_classification_project/notebooks/image_classification.ipynb`](./03_image_classification_project/notebooks/image_classification.ipynb)

---

## 4 Â· Spam Email Detection

**Problem:** Classify emails as spam or legitimate â€” the classic asymmetric-cost problem, where a spam email reaching the inbox is a mild annoyance but a legitimate email wrongly filtered could be a missed job offer.

### Pipeline

```
Raw email text
  â†“ Text preprocessing (NLTK) â€” cleaning, tokenization
  â†“ TF-IDF vectorization â€” weight terms by discriminative power
  â†“ Logistic Regression classifier
  â†“ Evaluate on held-out test set
```

### Results

| Split | Accuracy |
|-------|----------|
| Training | 96.11% |
| **Test** | **96.32%** |

> Test accuracy slightly *exceeding* training accuracy is a healthy sign â€” the model hasn't memorized the training set and generalizes to unseen emails. A large gap in the other direction would signal overfitting.

**Why TF-IDF over raw word counts:** it down-weights words appearing across all emails ("the", "and") and up-weights terms that actually distinguish spam from legitimate mail, so the classifier learns from signal rather than noise.

**How a stakeholder uses this:** an email filter tuned for the costlier error â€” a false positive hides a job offer, so thresholds favour keeping legitimate mail in the inbox.

ðŸ““ [`04_spam_email_detection_project/`](./04_spam_email_detection_project/)

---

## 5 Â· Sales Prediction

**Problem:** Predict product sales from outlet and item attributes â€” MRP, outlet size, item type, and outlet location. Sales forecasting drives inventory planning: under-forecast and you stock out; over-forecast and you tie up capital in unsold goods.

### Pipeline

```
Raw sales data
  â†“ Handle missing values and duplicates
  â†“ Exploratory data analysis (distribution, correlation)
  â†“ Encode categorical variables (outlet type, item category)
  â†“ Train Linear Regression
  â†“ Evaluate â€” MSE, MAE, RMSE, RÂ²
```

### Results

| Metric | Value |
|--------|-------|
| **RÂ² Score** | **0.91** |
| Mean Squared Error | 2.86 |

> RÂ² of 0.91 means the model explains 91% of the variance in sales. For a linear model on retail data that's strong â€” it suggests the relationship between MRP, outlet characteristics and sales is largely linear, so a more complex tree-based model would buy little here.

**How a stakeholder uses this:** inventory planners set stock depth per outlet â€” the model shows which outlet types justify deeper shelf allocation and which are structurally capped.

ðŸ““ [`05_sales_prediction_project/`](./05_sales_prediction_project/)

---

## 6 Â· Car Price Prediction

**Problem:** Estimate a used car's selling price from year, mileage, fuel type, transmission, and engine specs. Resale pricing is the core of any used-vehicle marketplace â€” mispricing costs either seller margin or buyer trust.

### Pipeline

```
Vehicle listings data
  â†“ Clean and inspect
  â†“ Feature engineering â€” derive vehicle age from year
  â†“ Encode categoricals (fuel type, transmission, seller type)
  â†“ Train Random Forest Regressor
  â†“ Evaluate â€” MAE, MSE, RÂ²
```

**Why Random Forest here:** depreciation isn't linear. A car loses value fastest in its first years and the curve flattens later; mileage interacts with age rather than acting independently. A tree ensemble captures those non-linear interactions without needing them specified in advance â€” which is precisely why it suits this problem where Linear Regression suited sales prediction.

**How a stakeholder uses this:** a marketplace pricing baseline â€” sellers get a defensible price band per vehicle instead of gut-feel quotes, reducing time-to-sale.

ðŸ““ [`06_car_price_prediction_project/`](./06_car_price_prediction_project/)

---

## 7 Â· Sentiment Analysis â€” Tweet Classification

**Problem:** Classify social media posts as positive, negative, or neutral â€” the core of brand monitoring and customer feedback analysis.

**Dataset:** 500 labelled tweets with text, sentiment, timestamp, and platform metadata.

### Pipeline

```
Raw text
  â†“ Lowercase  â†’  strip non-alphabetic characters
  â†“ Stopword removal (NLTK)  â†’  WordNet lemmatization
  â†“ TF-IDF vectorization (unigrams + bigrams, 5,000 features)
  â†“ Logistic Regression (class_weight='balanced', max_iter=1000)
```

Stratified 80/20 train-test split preserves class proportions.

### Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.85 | 0.41 | 0.55 | 27 |
| Neutral | 0.63 | 0.90 | 0.74 | 40 |
| Positive | 0.80 | 0.73 | 0.76 | 33 |
| **Accuracy** | | | **0.71** | 100 |
| **Weighted avg** | 0.75 | 0.71 | 0.70 | 100 |

> **Honest read:** the model is precise on negative tweets (0.85) but recalls only 41% of them â€” it misses many, usually labelling them neutral. With just 27 negative samples in the test set, more data is the clearest path to improvement. `class_weight='balanced'` was used specifically to counteract this imbalance.

**Includes** a `predict_sentiment()` helper for inference on unseen text.

**How a stakeholder uses this:** a brand-monitoring triage layer â€” positive/neutral posts pass through, likely-negative posts are routed to support for human review.

ðŸ““ [`07_sentiment_analysis_project/notebooks/sentiment_analysis.ipynb`](./07_sentiment_analysis_project/notebooks/sentiment_analysis.ipynb)

---

## 8 Â· Music Recommendation System

**Problem:** Given a song a listener enjoys, suggest similar tracks â€” the content-based approach used when you lack the user-interaction history that collaborative filtering requires.

**Dataset:** 100 songs with title, artist, genre, release year, duration, listen date, and platform.

### Approach

```
Raw listening data
  â†“ Deduplicate to unique songs
  â†“ Feature engineering â€” aggregate listen counts per song
  â†“ One-hot encode categorical features (genre, artist)
  â†“ StandardScaler normalization
  â†“ Cosine similarity matrix across all song pairs
  â†“ recommend_song(title, top_k) â†’ K nearest neighbours
```

**Why content-based:** it sidesteps the cold-start problem â€” a new song can be recommended immediately from its own attributes, with no listening history needed.

**How a stakeholder uses this:** cold-start-ready recommendations for a growing catalogue â€” every track is recommendable from day one, before any interaction history exists.

ðŸ““ [`08_music_recommendation_system/notebooks/music_recommendation.ipynb`](./08_music_recommendation_system/notebooks/music_recommendation.ipynb)

---

## Repository Structure

```
Data-Science-Project/
â”œâ”€â”€ 01_ride-demand-and-revenue-forecasting/# Forecasting Â· SARIMA + K-Means + RF fare model
â”œâ”€â”€ 02_subscription-retention-prediction/# Churn Â· LogReg + RF, modular pipeline with tests
â”‚   â”œâ”€â”€ src/                            # data_processing â†’ modeling â†’ explainability
â”‚   â””â”€â”€ Churn_Prediction_Analysis.ipynb
â”œâ”€â”€ 03_image_classification_project/       # CNN Â· MNIST digit recognition
â”‚   â””â”€â”€ notebooks/image_classification.ipynb
â”œâ”€â”€ 04_spam_email_detection_project/       # NLP Â· TF-IDF + Logistic Regression
â”œâ”€â”€ 05_sales_prediction_project/           # Regression Â· Linear Regression
â”œâ”€â”€ 06_car_price_prediction_project/       # Regression Â· Random Forest
â”œâ”€â”€ 07_sentiment_analysis_project/         # NLP Â· tweet sentiment classification
â”‚   â”œâ”€â”€ dataset/sentiment_analysis.csv
â”‚   â””â”€â”€ notebooks/sentiment_analysis.ipynb
â””â”€â”€ 08_music_recommendation_system/        # Content-based recommender
    â”œâ”€â”€ dataset/music_dataset.csv
    â””â”€â”€ notebooks/music_recommendation.ipynb
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://gitlab.com/ansht_121/Data-Science-Project.git
cd Data-Science-Project

# Install dependencies
pip install numpy pandas matplotlib scikit-learn tensorflow nltk jupyter

# Launch Jupyter
jupyter notebook
```

### Running Each Project

**Ride Demand Forecasting Â· Subscription Churn Prediction** â€” each folder ships its own `requirements.txt` and README.

```bash
pip install -r 01_ride-demand-and-revenue-forecasting/requirements.txt
pip install -r 02_subscription-retention-prediction/requirements.txt

jupyter notebook 01_ride-demand-and-revenue-forecasting/Forecast_Clustering_Analysis.ipynb
jupyter notebook 02_subscription-retention-prediction/Churn_Prediction_Analysis.ipynb
# or run the modular churn pipeline end-to-end:
python 02_subscription-retention-prediction/src/run_pipeline.py --help
```

**Image Classification** â€” no setup needed; MNIST downloads automatically on first run.

```bash
jupyter notebook 03_image_classification_project/notebooks/image_classification.ipynb
```

**Spam Detection Â· Sales Prediction Â· Car Price Prediction** â€” open the notebook in each project folder and run all cells.

```bash
jupyter notebook 04_spam_email_detection_project/
jupyter notebook 05_sales_prediction_project/
jupyter notebook 06_car_price_prediction_project/
```

> **Datasets not included.** These three notebooks expect `spam_mail.csv`, `sales dataset.csv`, and `car_data.csv` respectively, which aren't committed to this repository. Place your own copy alongside the notebook, or update the `read_csv()` path in the first cell. The notebooks retain their saved outputs, so the results are reproducible reading even without the raw files.

**Sentiment Analysis** â€” downloads NLTK corpora on first run.

```bash
cd 07_sentiment_analysis_project/notebooks
jupyter notebook sentiment_analysis.ipynb
```

> The notebook loads `sentiment_analysis.csv`. Either run it from the `dataset/` folder or update the path to `../dataset/sentiment_analysis.csv`.

**Music Recommendation**

```bash
cd 08_music_recommendation_system/notebooks
jupyter notebook music_recommendation.ipynb
```

> The notebook loads `MusicDataset.csv`. Update the path to `../dataset/music_dataset.csv` to match this repository's layout.

---

---

## ðŸ¤– How These Models Are Built in the AI Era

Every model in this repository follows the same discipline: **AI accelerates, humans validate.**

| Stage | AI Assists With | My Validation Gate |
| :--- | :--- | :--- |
| **Data prep** | Drafting cleaning & feature-engineering logic | Row-count reconciliation, dtype audits, leakage checks (e.g., SARIMA validated on held-out Novâ€“Dec, never on the training window) |
| **Modeling** | Baseline code scaffolding | Algorithm matched to problem structure â€” Linear Regression where the relationship is linear (RÂ² 0.91), Random Forest where it is not |
| **Evaluation** | Metric summaries | Metrics recomputed from scratch; train/test gaps interpreted, not hidden (sentiment model's weak minority-class recall is reported, not rounded up) |
| **Explainability** | Drafting interpretation text | Feature importances cross-checked against domain sense before being written into findings |

The principle: **AI is a co-pilot, not an oracle.** It drafts, I verify â€” a hallucinated evaluation metric or leaky validation split that *looks* right is more dangerous than an obvious failure. Model limitations are documented, not hidden, exactly because AI output can't be trusted on faith.

---

## Skills Demonstrated

| Area | Techniques |
|------|-----------|
| **Deep Learning** | CNN architecture design, convolution/pooling layers, softmax classification, Adam optimization |
| **NLP** | Text cleaning, stopword removal, lemmatization, TF-IDF with n-grams |
| **Classical ML** | Linear & logistic regression, Random Forest ensembles, class imbalance handling, stratified splitting |
| **Regression Analysis** | RÂ², MSE, MAE, RMSE interpretation; matching model complexity to relationship structure |
| **Recommender Systems** | Content-based filtering, one-hot encoding, cosine similarity, cold-start reasoning |
| **Time Series Forecasting** | SARIMA with weekly seasonality, held-out validation, Ljung-Box residual diagnostics, seasonal-naive baselines |
| **Unsupervised Learning** | K-Means demand segmentation, silhouette-based k selection, cluster profiling |
| **Churn Modeling** | Logistic Regression vs Random Forest, ROC-AUC / precision / recall trade-offs, explainability, risk scoring |
| **Evaluation** | Accuracy, precision, recall, F1-score, per-class analysis, train/test gap interpretation |
| **Data Handling** | Pandas transformations, feature scaling, deduplication, aggregation |

---

## Future Improvements

- **Sentiment Analysis** â€” expand beyond 500 samples; try SVM and transformer-based models (BERT) as baselines to beat
- **Image Classification** â€” add data augmentation and dropout; extend to Fashion-MNIST or CIFAR-10
- **Music Recommendation** â€” introduce collaborative filtering as a hybrid layer once interaction data is available
- **Across all** â€” refactor notebooks into reusable modules, add `requirements.txt`, and pin dependency versions

---

## Author

**Ansh Thakur** â€” Data Analyst & Machine Learning Practitioner

[![GitLab](https://img.shields.io/badge/GitLab-ansht__121-FC6D26?style=flat&logo=gitlab&logoColor=white)](https://gitlab.com/ansht_121)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Site-00C7B7?style=flat&logo=netlify&logoColor=white)](https://ansh-thakur-netlify.netlify.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ansh-thakur-5407192b4/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:ansht1194@gmail.com)
