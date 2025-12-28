# Cricket Score Prediction System
 ![WhatsApp Image 2025-12-18 at 9 55 13 PM](https://github.com/user-attachments/assets/04a80da1-d022-4322-8c8a-5648e14fb9d0)
 ![WhatsApp Image 2025-12-18 at 9 55 34 PM](https://github.com/user-attachments/assets/650fc729-2c8c-45a7-8d61-30b082678eab)
 ![WhatsApp Image 2025-12-18 at 9 56 23 PM](https://github.com/user-attachments/assets/a10f50e7-5a4e-4551-aa7b-6442cffe32fa)



## 📋 Project Overview
The **Cricket Score Prediction System** is an advanced machine learning application designed to forecast the final scores of cricket matches across multiple formats, including T20 International, One Day International (ODI), and Test matches. By analyzing current match variables—such as runs, wickets, overs, and historic match data—the system provides accurate, data-driven predictions in real-time.

Built with a sophisticated **XGBoost** regression model trained on extensive historical match datasets, this application serves as a powerful tool for cricket enthusiasts and analysts to gauge match outcomes with high precision.

## ✨ Key Features
*   **Multi-Format Support**: Seamless prediction capabilities for T20, ODI, and Test match formats.
*   **Advanced ML Architecture**: Powered by XGBoost for superior predictive accuracy and performance.
*   **Real-time Analysis**: Processes live match inputs (current score, wickets lost, overs bowled) to generate instant forecasts.
*   **Context-Aware**: Factors in critical elements such as venue (city), batting team, and bowling team strength.
*   **Premium User Interface**: Features a refined 'White Gold' aesthetic, offering a professional and engaging user experience.

## 🛠️ Technology Stack
*   **Backend Framework**: Flask (Python)
*   **Machine Learning**: XGBoost, Scikit-learn, Pandas, NumPy
*   **Frontend**: HTML5, CSS3 (Custom Styling)
*   **Deployment**: Docker / Cloud Ready

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/9mit/CRICKET_SCORE-PREDICTION-SYSTEM.git
cd CRICKET_SCORE-PREDICTION-SYSTEM
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Model Initialization
Before running the application for the first time, initialize the machine learning model:
```bash
python train_model.py
```
*Note: This will generate the `pipe.pkl` model file required for predictions.*

### Step 4: Launch Application
```bash
python app.py
```
Access the application dashboard at: `https://huggingface.co/spaces/Spidercraft01/cricket-score-predictor`

## 📊 Usage Guide
1.  **Select Match Format**: Choose between T20, ODI, or Test match.
2.  **Input Match Details**:
    *   Select Batting and Bowling teams from the standardized list.
    *   Choose the Host City.
    *   Enter current match statistics (Current Score, Overs Done, Wickets Fallen, Runs in Last 5 Overs).
3.  **Generate Prediction**: Click 'Predict Score' to receive the projected final total ranges.

## 🤝 Contributing
Contributions are welcome. Please ensure that pull requests include detailed descriptions of changes and maintain the existing coding standards.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
