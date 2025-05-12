# 🧠 Lightweight Object Detection with Faster R-CNN and Streamlit Demo

This project implements a **lightweight object detection system** using a **Faster R-CNN model** with a **MobileNetV2 backbone**, trained on a small subset of the **Pascal VOC 2007 dataset**. It includes a **Streamlit web app** for interactive image uploads and visualization of detection results.

---

## 🚀 Features

- ✅ Fast training using MobileNetV2 + Faster R-CNN
- 📦 VOC 2007 dataset support
- 🧠 Pretrained and custom-trained model options
- 🖼️ Bounding box visualization with class labels and scores
- 🌐 Streamlit-based interactive web interface

---

## 🗂️ Project Structure

├── train.py # Trains Faster R-CNN with MobileNetV2 backbone
├── test.py # Runs inference and draws bounding boxes
├── demo.py # Streamlit app for web-based testing
├── requirements.txt # Python dependencies
├── quick_rcnn_model.pth # (Generated after training) model weights
├── result_*.jpg # Output images after detection
├── VOCdevkit/ # Dataset folder (Pascal VOC 2007)

yaml
Copy
Edit

---

## 🧪 Quick Start

### 1. 📦 Install Requirements
```bash
pip install -r requirements.txt
2. 🏋️‍♂️ Train the Model (Optional)
bash
Copy
Edit
python train.py
Saves quick_rcnn_model.pth after training on a small VOC subset.

3. 🧠 Run Detection from Terminal
bash
Copy
Edit
python test.py img.jpg quick_rcnn_model.pth 0.3
4. 🌐 Launch Streamlit Web App
bash
Copy
Edit
streamlit run demo.py
Upload any .jpg, .png, or .jpeg file and see the results with bounding boxes and class names.

🧱 Model Architecture
Backbone: MobileNetV2 (frozen during training)

Head: Faster R-CNN detection head

RPN Anchors: Sizes (32, 64, 128), Aspect Ratios (0.5, 1.0, 2.0)

RoIAlign: Multi-scale, 7×7 pooling

📊 Dataset
Pascal VOC 2007

Automatically downloaded and processed via torchvision.datasets.VOCDetection

Only a small subset (20–50 images) is used for quick training

📷 Example Output
<img src="result_img.jpeg" alt="Result" width="600"/>
📚 Dependencies
torch

torchvision

Pillow

matplotlib

streamlit

(see requirements.txt)

🧠 AI Assistance Disclosure
Some parts of this project (e.g., data loading, model creation, Streamlit integration) were assisted using AI tools like ChatGPT for speed and debugging purposes. All AI-generated code was fully understood and adapted as necessary.

📬 Contact / Contributions
Pull requests and suggestions are welcome!
