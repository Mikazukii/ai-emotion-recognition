# Emotion Recognition Project

## Description

This project is an emotion recognition application based on a deep neural network. The goal is to predict human emotions from facial images using a machine learning model. The model has been trained using a labeled facial image dataset, and the application allows real-time emotion prediction from an image.

### Features:

- **Emotion Recognition**: The model predicts the following emotions: _Angry_, _Disgust_, _Fear_, _Happy_, _Sad_, _Surprise_, _Neutral_.
- **User Interface**: The application provides a graphical interface that allows the user to either take a live photo or import an image for prediction.
- **Privacy**: The images taken or imported are **not stored** by the algorithm. They are only used temporarily for prediction purposes.

## Project Structure

The project is structured as follows:

- `model_building.ipynb`:

  - This Jupyter Notebook explains the data analysis, model building, model improvement, and benchmarking. It provides a detailed overview of the model training process, including data preprocessing, neural network architecture, and model evaluation.

- `application_functions.py`:

  - This file contains the functions necessary for building the application that uses the emotion recognition model. The application allows users to take a live photo or import an image and predict the emotion associated with that image.

- `main.py`:
  - This file runs the entire project. It launches the application and orchestrates the usage of the model and other components needed to take or import an image, make a prediction, and display the results.

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

1. **Train the model**:

   - Open `model_building.ipynb` in Jupyter Notebook to understand the data analysis and the steps for building the model.
   - Follow the steps to train and evaluate the model.

2. **Run the application**:

   - To run the application, simply execute the `main.py` file:

   ```bash
   python main.py
   ```

   The application will allow you to choose between taking a live photo or importing an image for emotion prediction.

3. **Image Privacy**:
   - Be assured that the images taken or imported by the user are **only used for prediction purposes**. No images are stored or retained in the system. They are only temporarily processed for prediction.

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
