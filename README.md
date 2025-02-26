# **[Ficbot - Frontend & API](https://ficbotweb.com)**  

_An AI-powered Anime Character Generator for Aspiring Writers_

![Anime Character Generator](https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/gifs/namegen.gif)

---

## **📌 About the Project**

Ficbot is a machine learning–powered system designed to assist writers in creating original characters effortlessly. It leverages deep learning and NLP models to generate character names from images, with planned expansions for bio and image generation.

### **Project Structure**  
Ficbot is now split into two repositories for better organization:

- **[ficbot (this repository)](https://github.com/Pythonimous/ficbot)** – Contains:
  - The **frontend** (user interface).
  - The **API layer** that communicates with the backend inference service.
  - Docker configuration for deploying the combined frontend + API container.

- **[ficbot-backend](https://github.com/Pythonimous/ficbot-backend)** – Contains:
  - The **AI inference service** that processes images and generates names.
  - The **ML models** and related dependencies (TensorFlow, etc.).
  - Training scripts, dataset processing, and exploratory notebooks.

---

## **🖥 Technical Stack** 

- **Machine Learning & Inference:**  
  - **TensorFlow:** Powers the AI model used for generating character names from images.
  - **FastAPI:** Also used in the backend inference service for serving predictions.

- **Frontend & API:**  
  - **FastAPI:** Serves the API endpoints used by the frontend.  
  - **Bootstrap:** Provides a responsive and modern UI for the web interface.  
  - **HTML5/CSS3 & JavaScript:** Standard technologies for building interactive web applications.

- **Deployment & Infrastructure:**  
  - **Docker + AWS Lightsail:** A reliable and cost-effective VPS solution.

---

## 📊 Dataset & Exploratory Notebook  

Ficbot's AI models were trained using a **public dataset** of anime characters, which I compiled and explored in depth.  

🔹 **Dataset on Kaggle:** [Anime Character Dataset](http://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)  
🔹 **Exploratory Data Analysis Notebook:** [View on Kaggle](https://www.kaggle.com/your-kaggle-profile/notebook-link)  

This dataset includes **over 106,000 characters**, with names, bios, and images, making it a valuable resource for training NLP models.  

----

## **✨ Features**

### ✅ **Currently Available**
- **Image → Name Generator:**  
  Upload an image and get a character name based on AI analysis.

### 🚀 **Planned Enhancements**
- **Additional Name Generators:** (Based on bios and hybrid inputs)
- **Bio Generators:** (Generate detailed character backstories)
- **Image Generators:** (AI-generated character visuals)
- **Anime Filter:** (Transform images into an anime-style character)
- **Complete OC Generator:** (Generate Name, Bio, and Image together)

---

## **🛠 Installation**

### **1. Create and Activate a Virtual Environment**

**Windows (without WSL)**: [Guide](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html)  
**Linux / Windows (with WSL)**: [Guide](https://www.liquidweb.com/kb/how-to-setup-a-python-virtual-environment-on-windows-10/)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows

```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt

```

----------

## **🚀 Running the Application**

### Locally (Without Docker)

1. Ensure the backend inference service (ficbot-backend) is running.
2. Start the frontend + API server:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000

```

Once running, access the API at:  
📍 `http://127.0.0.1:8000/docs` (Interactive API Documentation)

> Note: The frontend API will need to communicate with the ficbot-backend service. Make sure the correct API URL is set in the frontend configuration.

## 🛠 Docker Deployment

This repository includes a Dockerfile for containerized deployment.

### 1️⃣ Build the Docker Image

```bash
docker build -t ficbot .

```

### 2️⃣ Run the Container

```bash
docker run -p 8000:8000 ficbot

```

Once running, you can access Ficbot's UI and API at your server's address.

----------

## 💂️ Testing & Development

### Running Unit Tests

```bash
python -m unittest
```

### Checking Test Coverage

```bash
pip install coverage
coverage run -m unittest
coverage report  # Current coverage: 92%
coverage html -d coverage_html # interactive html reporting

```

## **📌 Contributing**

We welcome contributions!

- Report issues or feature requests via GitHub Issues.
- Fork the repository and submit pull requests for new features or bug fixes.
- Check back for roadmap updates and community discussions.

----------

## **🐝 License**

This project is **open-source** under the BSD-3-Clause license.

----------

## **🔗 Links**

🔹 **Live Demo**: [ficbotweb.com](https://ficbotweb.com)  
🔹 **Ficbot Backend**: [ficbot-backend](https://github.com/Pythonimous/ficbot-backend)  
🔹 **Dataset**: [Kaggle](http://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)
