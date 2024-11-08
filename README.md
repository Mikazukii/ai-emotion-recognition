# Emotion Recognition Project

## Description

This project is an emotion recognition application based on a deep neural network. The goal is to predict human emotions from facial images using a machine learning model. The model has been trained using a labeled facial image dataset, and the application allows real-time emotion prediction from an image.

### Features:

- **Emotion Recognition**: The model predicts the following emotions: _Angry_, _Disgust_, _Fear_, _Happy_, _Sad_, _Surprise_, _Neutral_.
- **User Interface**: The application provides a graphical interface that allows the user to either take a live photo or import an image for prediction.
- **Privacy**: The images taken or imported are **not stored** by the algorithm. They are only used temporarily for prediction purposes.

## Project Structure

The project is organized as follows:

```
emotion-recognition/
├── __pycache__/
├── __init__.py
├── application_functions.py
├── model_building.ipynb
├── main.py
├── data/
│   └── fer20131.csv
├── models/
│   ├── augmented_facial_recognition_model.h5
│   ├── augmented_facial_recognition.h5
│   ├── facial_recognition_model.h5
│   └── facial_recognition.h5
```

### Detailed Structure:

- `model_building.ipynb`:

  - This Jupyter Notebook explains the data analysis, model building, improvement, and benchmarking processes. **Note**: The notebook contains computationally intensive scripts for model training. If you prefer not to spend time training the model, you can skip this step, as **pre-trained models are already available** in the `models` directory. Simply proceed to run the application using `main.py`.

- `application_functions.py`:

  - This file contains functions for the application that utilizes the emotion recognition model. The application allows users to take a live photo or import an image and predict the associated emotion.

- `main.py`:

  - This file executes the entire project. It launches the application, handles image input, runs the model prediction, and displays the results.

- `models/`:

  - This directory contains pre-trained models, which means you can directly use them without going through the model training process:
    - `augmented_facial_recognition_model.h5` and `augmented_facial_recognition.h5`: Enhanced models using data augmentation techniques.
    - `facial_recognition_model.h5` and `facial_recognition.h5`: Standard trained models.

- `data/`:
  - This folder contains the dataset used for training, including `fer20131.csv`, a popular dataset for facial emotion recognition.

## Installation

1. Clone this project to your local machine:

   ```bash
   git clone https://github.com/your-username/emotion-recognition.git
   ```

2. Install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Using Pre-trained Models

1. **Skip Training**:

   - Since pre-trained models are available in the `models` folder, you can directly run the application without training the model. This saves time and computational resources.

2. **Run the Application**:

   - Simply execute the `main.py` file:

   ```bash
   python main.py
   ```

   The application will allow you to choose between taking a live photo or importing an image for emotion prediction.

### Option 2: Training the Model (Optional)

1. **Train the Model**:
   - Open `model_building.ipynb` in Jupyter Notebook to review data analysis and training steps.
   - Follow the instructions to train and evaluate the model. This process is **time-consuming** due to intensive computations.

## Image Privacy

- Rest assured that the images taken or imported by the user are **only used for prediction purposes**. No images are stored or retained in the system. They are only temporarily processed for prediction.

## Technologies Used

- Python
- TensorFlow / Keras (for building the neural network model)
- OpenCV (for image handling)
- Tkinter (for the graphical interface)
- Matplotlib (for displaying results)

## Contributing

Contributions to this project are welcome. To suggest an improvement or report a bug, please follow these steps:

1. Fork this repository.
2. Create a branch for your feature (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Added a feature'`).
4. Push to your branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
