# **[Ficbot - Frontend & API](https://ficbotweb.com)**  

_An AI-powered Anime Character Generator for Aspiring Writers_

![Anime Character Generator](https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/gifs/namegen.gif)

---

## **📌 About the Project**

Ficbot is a machine learning–powered system designed to assist writers in creating original characters effortlessly. It leverages deep learning and NLP models to generate character names from images, with planned expansions for image generation.

### **Project Structure**  
Ficbot is now split into two repositories for better organization:

- **[ficbot (this repository)](https://github.com/Pythonimous/ficbot)** – Contains:
  - The **frontend** (user interface).
  - The **API layer** that communicates with the backend inference service.
  - Docker configuration for deploying the combined frontend + API container.

- **[ficbot-backend](https://github.com/Pythonimous/ficbot-backend)** – Contains:
  - The **AI inference service** that processes images and generates names and bios.
  - The **ML models** and related dependencies (Torch, Transformers).
  - Training scripts, dataset processing, and exploratory notebooks.


---
## **🖥 Technical Stack**

### 🔹 Machine Learning & Inference
- **PyTorch:** Powers the name generation model, which maps image embeddings to plausible character names.  
- **Transformers:** Used for bio generation with modern NLP architectures, fine-tuned on character metadata.  
- **Llama.cpp:** Enables fast, CPU-efficient inference of quantized language models for resource-constrained environments.  
- **FastAPI:** Serves the backend inference API and provides a lightweight, async-compatible framework for handling requests.

### 🔹 Frontend & Interface
- **HTML5/CSS3 & JavaScript:** Core technologies for the interactive user interface.  
- **Bootstrap:** Ensures responsive, mobile-friendly layouts.  
- **FastAPI (Frontend API layer):** Also serves static assets and API endpoints consumed by the frontend.

### 🧠 RAG-Inspired Pipeline
Character names are vectorized and compared to the dataset of top-1000 most popular characters from MyAnimeList.net. The top-matching entries are used to extract genres and themes, which then guide bio generation. This retrieval-enhanced approach grounds outputs in contextually relevant tropes and improves narrative coherence.

## 📊 Dataset & Exploratory Notebook  

Ficbot's AI models were trained using a **public dataset** of anime characters, which I compiled and explored in depth.  

🔹 **Dataset on Kaggle:** [MyAnimeList Character Dataset](https://www.kaggle.com/datasets/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)  
🔹 **Exploratory Data Analysis Notebook:** [View on Kaggle](https://www.kaggle.com/code/ophelion/myanimelist-dataset-exploratory-notebook)  

This dataset includes **over 106,000 characters**, with names, bios, and images, making it a valuable resource for training NLP models.  

---

## **✨ Features**

### ✅ **Currently Available**

- **🔹 Image → Name Generator**  
  Upload an image, and the AI will generate a character name based on visual analysis.  
  ✨ *Powered by MobileNetV3 + Bidirectional LSTM.*
- **🔹 Bio Generator**  
  Provide a name, and the AI will generate a detailed character bio.  
  ✨ *Built on DistilGPT2 with sliding window chunking (50–200 tokens) and optimized for CPU inference via LlamaCPP. Uses a lightweight RAG-inspired approach: vectorizes the input name, retrieves the most similar characters from the popular characters' set, and injects their anime genres and themes to ground the output in believable character types.*
- **🔹 Anime Filter**
  With a press of a button (**stylize**), transforms your image into an anime style!
  ✨ *Uses Bryandlee's PyTorch implementation of AnimeGAN2. [Check it out!](https://github.com/bryandlee/animegan2-pytorch)*

---

### 🚀 **Planned Enhancements**
- **🔹 Advanced Name Generator** – Enable name generation based on bios or hybrid inputs for more coherent character design.
- **🔹 Customizable Themes & Genres** – Allow users to manually input preferred themes or genres (e.g., action, pirates, magical girl) to influence bio generation, instead of relying solely on automatic RAG retrieval.
- **🔹 AI-Powered Image Generation** – Generate character portraits based on the outputted name and bio using image synthesis models.
- **🔹 Complete OC Generator** – One-click generation of complete original characters, including name, bio, and image.

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
📍 `https://127.0.0.1:8000/docs` (Interactive API Documentation)

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

## Acknowledgements

This project utilizes [AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch) for anime-style image transformation.

- **AnimeGAN2-PyTorch** by [bryandlee](https://github.com/bryandlee)  
  - Repository: [AnimeGAN2-PyTorch](https://github.com/bryandlee/animegan2-pytorch)
  - License: [MIT](https://opensource.org/licenses/MIT).

----------

## **🔗 Links**

🔹 **Live Demo**: [ficbotweb.com](https://ficbotweb.com)  
🔹 **Ficbot Backend**: [ficbot-backend](https://github.com/Pythonimous/ficbot-backend)  
🔹 **Dataset**: [Kaggle](https://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)  
🔹 **Exploratory Data Analysis Notebook:** [Kaggle](https://www.kaggle.com/code/ophelion/myanimelist-dataset-exploratory-notebook)  
