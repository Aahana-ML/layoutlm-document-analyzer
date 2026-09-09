# 📄 LayoutLM Document Analyzer

A document understanding system built using **PaddleOCR** and **LayoutLM** to detect and classify important elements in document images.

The project combines **text, bounding-box coordinates, and document layout information** to perform token-level document classification.

## 🚀 Live Demo

Try the deployed application:

https://layoutlm-document-analyzer-dewxr3sirrxkrj6qwdqpqz.streamlit.app/

## 🤗 Model

The trained LayoutLM model is hosted on Hugging Face:

https://huggingface.co/2-13aahana/layoutlm-document-analyzer

## 📌 Project Overview

The goal of this project is to build a document understanding pipeline that can identify different types of content inside a document.

The system first extracts text and its spatial coordinates using PaddleOCR. These coordinates are then converted into the format required by LayoutLM. The trained LayoutLM model predicts a label for each detected word.

The predictions are finally visualized on the original document using colored bounding boxes.

## 🔄 Pipeline

Document Image
        ↓
PaddleOCR
        ↓
Words + Bounding Boxes
        ↓
Original Pixel Coordinates
        ↓
Coordinate Normalization
        ↓
LayoutLM Tokenization
        ↓
Trained LayoutLM Model
        ↓
Token-Level Predictions
        ↓
Label Mapping
        ↓
Bounding Box Visualization

## 🧠 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- LayoutLM
- PaddleOCR
- PaddlePaddle
- Streamlit
- Hugging Face Hub

## 🏷️ Document Labels

The model currently groups document elements into three main categories:

- **QUESTION**
- **ANSWER**
- **HEADER**

Each detected element is displayed using a different bounding-box color to make the predictions easier to interpret visually.

## 🎨 Visualization

The application displays the model's predictions directly on the document.

Each predicted element contains:

- A bounding box
- Its predicted label
- A label color corresponding to its class

This makes it easier to visually inspect how the model understands the document layout.

## 📂 Project Structure

```text
layoutlm-document-analyzer/
│
├── app.py
├── inference.py
├── requirements.txt
├── README.md
├── .gitignore
└── test_image.png
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/layoutlm-document-analyzer.git
cd layoutlm-document-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## ⚠️ Limitations
This is a trained document-understanding model and its predictions are not guaranteed to be correct.

In particular, the model may incorrectly classify some document elements as QUESTION, ANSWER, or HEADER.

The visualization therefore represents model predictions, rather than guaranteed document annotations.

---

## 🔮 Future Improvements

Future versions of the project can extend the current document understanding pipeline with:

 - Question-answer extraction
 - Question-to-answer matching
 - Natural-language document queries
 - Improved classification accuracy
 - Better handling of complex document layouts
 - More document categories
 - Improved inference speed

---

## 🤝 Acknowledgements

This project was developed as an AI/ML learning project with guidance and learning support from **ChatGPT**.

---

## 👩‍💻 Author

**Aahana**

GitHub: [Aahana-ML](https://github.com/Aahana-ML)
