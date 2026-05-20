# PPE Detection & Safety Monitoring System

## Overview
An AI-powered PPE (Personal Protective Equipment) detection and construction safety monitoring system built using YOLOv8, Streamlit, and computer vision techniques.

The system detects safety violations such as missing helmets, masks, and safety vests from real-time images and videos while providing analytics dashboards and AI-based safety assistance.

---

## Features
- Real-time PPE detection
- Helmet, mask, and safety vest detection
- Safety violation monitoring
- Live video and image inference
- Interactive Streamlit dashboard
- PPE compliance analytics
- Detection metrics visualization
- AI Safety Assistant using Groq LLM
- Upload and analyze custom videos/images
- Automated quality evaluation workflow
- DVC-based metrics tracking

---

## Tech Stack
- Python
- YOLOv8
- OpenCV
- Streamlit
- Plotly
- NumPy
- Pandas
- PyTorch
- Ultralytics
- Groq LLM
- DVC

---

## System Workflow
1. Collect and preprocess PPE dataset
2. Train YOLOv8 detection model
3. Run image/video inference
4. Detect PPE compliance violations
5. Generate detection analytics and metrics
6. Display dashboard visualizations
7. Provide AI-based safety assistance

---

## Key Highlights
- Achieved 90% detection accuracy
- Real-time image and video processing
- Automated PPE compliance monitoring
- Interactive analytics dashboard
- AI-powered chatbot for safety insights
- Supports multiple PPE violation classes

---

## Project Structure

```text
ppe-detection-safety-monitoring-system/
│
├── screenshots/
├── metrics/
├── src/
├── result/
│
├── app.py
├── chatbot.py
├── detect.py
├── train.py
├── requirements.txt
├── params.yaml
├── dataset.yaml
├── dvc.yaml
├── dvc.lock
├── runs.dvc
├── README.md
├── .gitignore
└── .dvcignore
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Run Project

```bash
streamlit run app.py
```

---

## Sample Detection Output

### Detection Result
Add:
```text
screenshots/detection-output.jpg
```

### Dashboard Preview
Add:
```text
screenshots/dashboard.png
```

### Chatbot Preview
Add:
```text
screenshots/chatbot.png
```

---

## Future Enhancements
- CCTV integration
- Cloud deployment
- Multi-camera monitoring
- Real-time safety alerts
- Email/SMS notification system

---

## Author
Unnati Lunawat