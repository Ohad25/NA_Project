## Project Goal

This project aims to classify EEG signals collected during motor imagery tasks. The goal is to differentiate between trials where a subject imagined moving their left or right hand. The pipeline extracts informative features from raw EEG, applies dimensionality reduction, and trains classifiers to predict motor intention.

---

## Dataset Description

- **Training Data:** `motor_imagery_train_data.npy` 
- Shape: `(128 trials × 768 time samples × 3 channels)` 
- **Test Data:** `motor_imagery_test_data.npy`
- **Predictions:** `y_pred.npy` (model's output predictions)

---

## Project Structure

NA_Project:
- `classification.py`: Model selection and evaluation
- `features.py`: Feature extraction and PCA
- `final_project_script_scaffold.py`: Main entry point and Test
- `motor_imagery_dataset.py`: Dataset wrapper class
- `motor_imagery_train_data.npy`: Training EEG data
- `motor_imagery_test_data.npy`: Test EEG data (unlabeled)
- `power_spectra.py`: Power spectrum & spectrogram analysis
- `visualization.py`: EEG visualization
- `y_pred.npy`: Final predictions output
- `README.md`: This file
---

## Architecture & Pipeline

This project follows the pipeline below:

The main function resides in `final_project_script_scaffold.py`. It integrates all major components:

1. **Data Loading** via `motor_imagery_dataset.py`

2. **Visualization** via `visualization.py`
   - Plot raw EEG signals
   - Plot EEG

3. **Power Spectra and Spectograms** via `power_spectra.py`
   - Compute Power Spectral Density (PSD)
   - Calculate band power (alpha, beta, etc.)
   - Plot spectograms

4. **Feature Extraction And Dimensionality Reduction** via `features.py`
   - Frequency-domain features (band power)
   - Time-domain features (RMS, variance)
   - Entropy features (spectral entropy)
   - Combine into feature matrix per trial
   - Apply PCA
   - Visualize using PCA plot

5. **Classification** via `classification.py`
   - Train LDA classifier
   - Cross-validate on training data using 10-fold cross-validation
   - Explore dimensionality reduction methods (PCA, ANOVA, RFE)
   - Grid search for best pipeline
   - Best method and number of features selected

6. **Model Training & Testing** via the main function in `final_project_script_scaffold.py`
   - The best-performing model is trained on the full training set
   - This trained model is returned and used to transform and predict the **test dataset**
   - Test predictions are saved to `y_pred.npy`

---

## Requirements

- matplotlib==3.10.3
- numpy==2.3.1
- pandas==2.3.0
- scikit_learn==1.7.0
- scipy==1.16.0
