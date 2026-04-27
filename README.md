# Pneumonia CNN Project

This project trains a convolutional neural network (CNN) with TensorFlow to classify chest X-ray images as `NORMAL` or `PNEUMONIA`.

## Project Structure

```text
pneumonia_cnn_project/
├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
├── model.py
├── README.md
└── .gitignore
```

## Requirements

- Python 3.11 or 3.12 recommended
- TensorFlow
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install tensorflow matplotlib scikit-learn
```

On this machine, TensorFlow was run successfully with Python 3.11.

## Dataset

This repository does not include the `chest_xray` dataset because it is too large for a normal GitHub repository.

After downloading and extracting the dataset, place it like this:

```text
chest_xray/
├── train/
├── val/
└── test/
```

Then move `chest_xray` into the project folder:

```text
pneumonia_cnn_project/
├── chest_xray/
└── model.py
```

## Training

Run the training script from inside the project folder:

```bash
python model.py
```

If your default Python version is not compatible with TensorFlow, use a compatible interpreter directly. Example on Windows:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" model.py
```

## Output

After training, the script will:

- print the test accuracy
- save the trained model as `cnn_pneumonia_model.h5`

Example output:

```text
Test accuracy: 0.8060897588729858
```

## Notes

- The current script trains for 5 epochs.
- The generated `.h5` model file is also ignored in git because it is a build artifact.
