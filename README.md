# **[Ficbot](https://ficbotweb.com/generate)**  

_An AI-powered Anime Character Generator for Fan Fiction Writers_

![Anime Character Generator](https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/gifs/namegen.gif)

## **📌 About the Project**

Ficbot is a **machine learning-based system** designed to help aspiring writers create characters effortlessly. It leverages **deep learning and NLP** to generate names (bios + images planned) for original characters (OCs).

This project includes:  
🇽 **Backend (TensorFlow, FastAPI)**: Handles AI model inference, data processing, and API endpoints.  
🇽 **Frontend (Bootstrap)**: Provides a web-based interface for interactive character generation.


👉 The project originated from **anime character data** on [MyAnimeList](https://myanimelist.net/) and was later expanded for more creative writing applications.

----------

## **✨ Features**

### ✅ **Currently Available**

-   **Image → Name Generator**: Upload an image, and the model suggests a fitting name for your character.

### 🚀 **Planned Features**

-   **Name Generators** (From Bio, Image + Bio).
-   **Bio Generators** (From Name, Image).
-   **Image Generators** (From scratch, Name, Bio).
-   **Anime Filter** (Turn yourself into an anime-style OC!).
-   **Complete OC Generator** (Generate Name, Bio, and Image together).

----------

## **🛠 Installation**

### **1⃣ Create and Activate a Virtual Environment**

**Windows (without WSL)**: [Guide](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html)  
**Linux / Windows (with WSL)**: [Guide](https://www.liquidweb.com/kb/how-to-setup-a-python-virtual-environment-on-windows-10/)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows

```

### **2⃣ Install Dependencies**

```bash
pip install -r requirements.txt

```

----------

## **🚀 Running the Application**

```bash
uvicorn src.api.main:app --reload

```

Once running, access the API at:  
📍 `http://127.0.0.1:8000/docs` (Interactive API Documentation)


----------

## **💂️ Dataset**

The dataset was crawled from **MyAnimeList.net** using Selenium and the [Jikan API](https://jikan.moe/).  
**Raw dataset** is available here: [📂 Kaggle Dataset](http://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353).

You can redownload it using the `download.py` script:

```bash
python src/data/download.py

```

----------

## **🛠 Development & Testing**

### **Running Unit Tests**

```bash
python -m unittest
```

### **Checking Test Coverage**

```bash
pip install coverage
coverage run -m unittest
coverage report  # Current coverage: 79%

```

----------

## **📌 Contributing**

We welcome contributions!

-   Report issues on **GitHub Issues**.
-   Fork & submit PRs for new features.
-   Stay tuned for roadmap updates!

----------

## **🐝 License**

This project is **open-source** under the BSD-3-Clause license.

----------

## **🔗 Links**

🔹 **Live Demo**: [ficbotweb.com](https://ficbotweb.com/generate)  
🔹 **Dataset**: [Kaggle](http://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)

