#  Movie Recommendation System

A Content-Based Movie Recommendation System built using **Machine Learning**, **Natural Language Processing (NLP)**, **FastAPI**, and **Streamlit**. The application recommends movies based on content similarity and enriches recommendations with real-time movie posters and metadata from the TMDB API.

##  Features

* Content-based movie recommendations
* NLP-powered text preprocessing
* TF-IDF Vectorization for feature extraction
* Cosine Similarity for recommendation generation
* Real-time movie posters and metadata using TMDB API
* FastAPI backend for serving recommendations
* Interactive Streamlit frontend
* Production-ready deployment architecture

---

##  Tech Stack

### Machine Learning & NLP

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### External API

* TMDB API

### Deployment

* Render (Backend)
* Streamlit Cloud (Frontend)

---

##  Dataset

The project uses the TMDB Movies Dataset containing approximately **45,000 movies**.

Dataset includes:

* Movie Titles
* Overviews
* Genres
* Taglines
* Metadata

---

##  Project Workflow

### 1. Data Cleaning & Preprocessing

* Removed duplicate records
* Handled missing values
* Selected relevant movie attributes
* Created a unified text feature (`tags`) by combining:

  * Overview
  * Genres
  * Tagline

### 2. NLP Processing

The text data was processed using NLP techniques:

* Lowercasing
* Punctuation removal
* Stopword removal
* Text normalization

This helps improve recommendation quality and reduce noise.

### 3. TF-IDF Vectorization

Text data is transformed into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

This allows machine learning algorithms to understand textual information mathematically.

### 4. Cosine Similarity

The recommendation engine uses **Cosine Similarity** to measure the similarity between movies.

Movies with the highest similarity scores are returned as recommendations.

### 5. Model Serialization

The trained vectorizer and similarity components are saved using:

* Pickle (.pkl)

This enables faster loading during deployment.

### 6. API Development

A FastAPI backend was built to:

* Load trained models
* Process recommendation requests
* Return recommended movies
* Fetch movie metadata from TMDB

### 7. Frontend Development

A Streamlit application provides:

* Movie search interface
* Recommendation display
* Movie posters
* Additional movie information

---

##  Project Structure

```bash
Movie_Recommendation_System/
│
├── DataSet/
│   ├── movies.csv
│  
│
├── Models/
│   ├── df.pkl
│   └── indices.pkl
    └── tfidf_matrix.pkl
    └── tfidf.pkl
│
├── Backend/
│   ├── main.py
│   
│
├── Frontend/
│   ├── app.py
│  
│
├── Notebooks/
│   └── movies.ipynb
│
└── README.md
```

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/harshwadari/Movie_Recommendation_System
cd Movie_Recommendation_System
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  TMDB API Setup

1. Create an account on TMDB.
2. Generate an API key.
3. Add the API key to your environment variables.

```env
TMDB_API_KEY=your_api_key_here
```

---

##  Run Backend

```bash
uvicorn Backend.main:app --reload
```

Backend will be available at:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

##  Run Frontend

```bash
streamlit run Frontend/app.py
```

Frontend will open in your browser automatically.

---

##  Future Improvements

* Hybrid Recommendation System
* Collaborative Filtering
* Deep Learning-Based Recommendations
* User Authentication
* Watchlist Feature
* Personalized Recommendations
* Movie Reviews and Ratings Integration

---

##  Key Learnings

Through this project, I gained practical experience in building and deploying an end-to-end Machine Learning application.

### Machine Learning

* Understanding content-based recommendation systems.
* Building recommendation engines using similarity metrics.
* Working with large-scale movie datasets.
* Model serialization using Pickle.

### Natural Language Processing (NLP)

* Text preprocessing and cleaning.
* Stopword removal and punctuation handling.
* Feature engineering using movie metadata.
* Converting textual data into numerical representations using TF-IDF.

### Mathematics Behind Recommendations

* Understanding vector space representations.
* Implementing and applying Cosine Similarity.
* Measuring semantic similarity between documents.

### Backend Development

* Building REST APIs using FastAPI.
* API routing and request handling.
* Model serving for machine learning applications.
* Interactive API documentation using Swagger UI.

### Frontend Development

* Creating interactive user interfaces with Streamlit.
* Integrating backend APIs with frontend applications.
* Displaying dynamic movie recommendations and metadata.

### Deployment & MLOps

* Deploying machine learning applications to production.
* Hosting backend services on Render.
* Deploying frontend applications on Streamlit Cloud.
* Managing environment variables and API keys securely.

### External API Integration

* Consuming third-party APIs (TMDB API).
* Fetching real-time movie posters and metadata.
* Handling API responses and data transformation.

### Software Engineering Skills

* Project structuring and modular development.
* Version control using Git and GitHub.
* Debugging and troubleshooting production issues.
* Building scalable and maintainable applications.


---

##  Acknowledgements

* TMDB for movie metadata and posters
* Scikit-Learn for machine learning tools
* FastAPI for backend development
* Streamlit for frontend development

---

## 📜 License

This project is intended for educational and portfolio purposes.
