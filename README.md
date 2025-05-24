# 🚗 Claim Intent Prediction

This project predicts whether an insurance customer is likely to make a claim based on vehicle and customer information. It uses a Random Forest Classifier trained on a preprocessed dataset and provides a user-friendly interface via **Streamlit**.

## 🌟 Features

- Predicts insurance claim intent
- Clean and interactive UI built with Streamlit
- Simple model training using scikit-learn
- Trained on key features like vehicle age, fuel type, and region

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/your-username/claim-intent-predictor.git
cd claim-intent-predictor
```

### Install dependencies
```bash

pip install -r requirements.txt
```
### Run the Streamlit app
```bash

streamlit run app.py
```
📁 Project Structure
```bash
claim-intent-predictor/
│
├── app.py                # Streamlit frontend
├── model.py              # Model training script
├── model.pkl             # Trained ML model
├── encoder.pkl           # Label encoder for categorical features
├── Insurance claims data.csv  # Dataset
├── requirements.txt
├── README.md
└── LICENSE
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
