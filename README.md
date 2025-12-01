---

# 💻 Laptop Price Predictor (Machine Learning + Streamlit)

Predict the price of any laptop based on specifications such as brand, CPU, RAM, GPU, storage, and more.
This project uses **Linear Regression**, **Feature Engineering**, **Log Transformation**, and a **Streamlit Web Interface** to provide real-time laptop price estimates.

---

## 🚀 Features

### 🔍 **Machine Learning**

* Linear Regression model
* Trained on real laptop dataset(`predictor_model\dataset\laptopData.csv`)
* Log-transformation applied on target variable
* Feature encoding (Label Encoding + One-Hot Encoding)
* Scaled numerical features for stable prediction

### 🌐 **Streamlit Web App**

* Clean, modern dark UI
* Background image support
* Blurred overlay for premium look
* Dropdowns, number inputs, select boxes
* Real-time price prediction

### 📦 **Model Capabilities**

* Predicts laptop price with good accuracy
* Handles multiple brands, CPUs, GPUs, RAM sizes
* Works for mid-range + high-end laptops
* Can be extended to new features anytime

---

## 🛠️ Tech Stack

| Technology               | Purpose                    |
| ------------------------ | -------------------------- |
| **Python**               | Model development          |
| **Pandas, NumPy**        | Data cleaning & processing |
| **Scikit-learn**         | ML model                   |
| **Streamlit**            | UI & Deployment            |
| **Matplotlib / Seaborn** | Exploratory Data Analysis  |
| **Pickle**               | Save model pipeline        |

---

## 📁 Project Structure

```
Laptop-Price-Predictor/
│
├── frontend/                      
│     └── app.py              # Streamlit UI  
│
├── model/                    
│     ├── pipeline.pkl        # Saved ML model  
│     └── model.ipynb         # Training notebook  
│
├── data/                     
│     └── clean_data.csv     # Clean Dataset   
│     └── final_data.csv     # Final Dataset
│     └── LaptopData.csv     # Foundation data(Unclean data) (if included)
│     └── training_data.csv  # Training data
│
├── README.md                 
└── requirements.txt          
```

---

## 📊 Dataset Description

The model is trained on real laptop specifications containing:

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| Company          | Laptop brand                          |
| TypeName         | Gaming / Ultrabook / Notebook .etc    |
| Ram              | Memory size                           |
| Weight           | Weight of laptop                      |
| Cpu brand        | Processor type                        |
| Gpu brand        | Graphics card                         |
| HDD/SSD          | Storage configuration                 |
| IPS, TouchScreen | Display features                      |
| PPI              | Pixel density                         |
| Price            | Target variable (logged)              |

---

## ⚙️ How It Works

### 1️⃣ Data Cleaning

* Remove duplicates & missing data
* Clean text columns
* Convert categorical → numerical

### 2️⃣ Feature Engineering

* PPI (pixel density) calculated
* Touchscreen → 0/1
* IPS → 0/1

### 3️⃣ Log Transformation

```python
y = np.log(df["Price"])
```

### 4️⃣ Other Model that i use in this project
* Lasso
* Ridge
- Both perform similiarly


### 4️⃣ Pipeline

```python
l_model = LinearRegression()
lr_pipe =Pipeline([
            ('step_1',encoding),
            ('step_2',l_model)
])
```

### 5️⃣ Streamlit Prediction

```python
predicted_price = np.exp(pipe.predict(query)).round(2)
```

---

## 🎨 Streamlit UI (Dark Theme + Blur + Image)

Your app includes:

✔ Black theme
✔ Blurred overlay
✔ Wallpaper background
✔ Organized 3-column input layout
✔ Modern prediction card

---

## ▶️ Run the Project

### Install dependencies

```
pip install -r requirements.txt
```

### Run Streamlit app

```
streamlit run frontend/app.py
```

---

## 📈 Model Performance

* Works extremely well on most brands
* Good accuracy (±5–10%)
* Slightly weak on Apple processors (not included in training)
* R2 Score(~0.87)

---

## 🔮 Future Improvements

* Add support for Apple M-series CPUs
* Use RandomForest or XGBoost for better accuracy
* Deploy online with Streamlit Cloud
* Add image upload → predict specs from image

---

## 🤝 Contributing

Feel free to fork the repo and improve model accuracy or UI.

---

## ⭐ If you like this project

Give it a **star ⭐ on GitHub** to support!

---

