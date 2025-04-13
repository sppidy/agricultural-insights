# Agricultural Insights - Crop and Yield Analysis

A modern web application for agricultural insights that includes crop recommendation and rice yield prediction using scikit-learn machine learning models.

## Features

- **Crop Recommendation**: Get personalized crop recommendations based on soil parameters (N, P, K, pH, etc.)
- **Yield Prediction**: Analyze important factors affecting rice yield in different states
- **Natural Language Q&A**: Ask agricultural questions in natural language

## Project Structure

```
.
├── docker-compose.yml
├── README.md
├── backend/
│   ├── app.py                # Main Flask application
│   ├── dataset_dl.py         # Script for downloading datasets (if applicable)
│   ├── Dockerfile            # Backend Dockerfile
│   ├── requirements.txt      # Python dependencies
│   ├── train.py              # ML model training script
│   ├── datasets/             # Agricultural datasets
│   │   ├── crop_data.csv
│   │   ├── Crop_Recommendation.csv
│   │   └── ICRISAT-District Level Data.csv
│   └── models/               # Trained ML models and artifacts
│       ├── crop_label_encoder.pkl
│       ├── crop_recommendation_scaler.pkl
│       ├── crop_recommendation_sklearn_model.pkl
│       ├── crop_unique_values.pkl
│       ├── dataset_metadata.pkl
│       ├── rice_yield_sklearn_model.pkl
│       ├── yield_categorical_encoders.pkl
│       ├── yield_categorical_features.pkl
│       ├── yield_feature_importance.pkl
│       ├── yield_numerical_features.pkl
│       └── yield_numerical_scaler.pkl
└── frontend/                 # React/Vite frontend
    ├── Dockerfile            # Frontend Dockerfile
    ├── eslint.config.js      # ESLint configuration
    ├── index.html            # Main HTML file
    ├── nginx.conf            # Nginx configuration
    ├── package.json          # Node.js dependencies and scripts
    ├── postcss.config.js     # PostCSS configuration
    ├── README.md             # Frontend specific README
    ├── tailwind.config.js    # Tailwind CSS configuration
    ├── vite.config.js        # Vite configuration
    ├── public/               # Static assets served directly
    │   └── vite.svg
    └── src/                  # React source code
        ├── App.css
        ├── App.jsx           # Main React application component
        ├── index.css
        ├── main.jsx          # Entry point for React app
        └── assets/           # Frontend assets (images, etc.)
            └── react.svg
```

## Technologies Used

- **Backend**: Python, Flask, scikit-learn, pandas, numpy
- **Frontend**: React, Tailwind CSS, Axios
- **Deployment**: Docker, Nginx

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Git

### Setup and Running

1. Clone the repository
   ```bash
   git clone https://github.com/sppidy/agricultural-insights.git
   cd agricultural-insights
   ```

2. Build and run using Docker Compose
   ```bash
   docker-compose up --build
   ```

3. Access the application
   - Frontend: http://localhost:8101
   - Backend API: http://localhost:5010/api

## API Endpoints

- `GET /api/health`: Health check endpoint
- `GET /api/info`: Get information about available models and data
- `POST /api/ask`: Process natural language questions
- `POST /api/recommend-crop`: Crop recommendation endpoint
- `POST /api/predict-yield`: Rice yield prediction endpoint
- `POST /api/combined-analysis`: Unified analysis endpoint

## Backend Development

If you want to retrain the models:

```bash
cd backend
python train.py
```

## Frontend Development

For local frontend development:

```bash
cd frontend
npm install
npm run dev
```