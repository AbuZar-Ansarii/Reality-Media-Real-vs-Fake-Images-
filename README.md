# 🖼️ AI-Powered Real vs Fake Image Feed

A minimal Pinterest-style image feed built with **Streamlit** that automatically analyzes images using a **TensorFlow deep learning model** to classify them as **Real or Fake**.

Images can be added manually by placing them in a folder or uploaded directly through the app’s sidebar. Every image is processed by the trained model and displayed with its prediction and confidence score.

<img width="1920" height="1080" alt="Screenshot (97)" src="https://github.com/user-attachments/assets/9602acb5-4e82-47c5-8ac6-99a4aec3d2bc" />
<img width="1920" height="1080" alt="Screenshot (98)" src="https://github.com/user-attachments/assets/eb278f16-092b-4777-be05-5d423f645218" />
<img width="1920" height="1080" alt="Screenshot (94)" src="https://github.com/user-attachments/assets/7acb7c2c-713d-4263-b18f-fa1039ffd852" />
<img width="1920" height="1080" alt="Screenshot (93)" src="https://github.com/user-attachments/assets/096f6565-fec7-4edc-a339-606f6f07242f" />

---

## 🚀 Features

- 📂 Auto-load images from a directory
- 📤 Sidebar image upload (single or multiple)
- 🔀 Shuffled Pinterest-style image feed
- 🧠 Real vs Fake image classification using TensorFlow
- 📊 Confidence score displayed with each image
- ⚡ Model caching for fast performance
- 🖥️ Clean, minimal UI

---

## 🧠 Machine Learning Model

- **Framework:** TensorFlow / Keras
- **Task:** Binary image classification (Real vs Fake)
- **Input Size:** 256 × 256
- **Output:** Class label + confidence score
- **Model File:** `fake_image_detector_model.h5`


```bash
## 🗂️ Project Structure

image-feed-app/
│
├── app.py
├── fake_image_detector_model.h5
├── feed_images/
│ ├── sample1.jpg
│ ├── sample2.png
│
├── requirements.txt
└── README.md

git clone https://github.com/AbuZar-Ansarii/Reality-Media-Real-vs-Fake-Images-.git
cd real-vs-fake-image-feed

