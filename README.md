# Explainable AI for Alzheimer's Diagnosis: Textual and Visual Interpretations from MRI

For this project you need to download the following project on the folder and, in case, adjust the import:
[Explainable Alzheimer's Disease Detection from MRI Scans](https://github.com/katyatrufanova/explainable-alzheimers-disease-detection)

## Project Overview

This project implements a Vision Transformer (ViT) model for image captioning, specifically designed for Alzheimer's disease diagnosis using MRI data. The model generates textual reports from MRI scans, providing both visual and textual explanations to assist medical professionals in understanding the AI's decisions. The visual explainability is achieved through Grad-CAM, highlighting relevant regions in the MRI, while the textual component generates detailed clinical reports explaining the model’s predictions.

## Key Features

- **Artificial Dataset**: The training dataset includes MRI images and artificial textual reports, as real clinical reports were unavailable.
- **ViT Architecture**: Utilizes a Vision Transformer model for the image captioning task.
- **Image and Textual Explainability**: Generates both visual (Grad-CAM) and textual (clinical report) explanations for Alzheimer’s diagnosis.

## Requirements

To run the project, you must first install all the dependencies listed in the `requirements.txt` file. These include necessary libraries for deep learning, image processing, and text generation.

### Installing Requirements

```bash
pip install -r requirements.txt
```

## Results

The model will output:

- **Textual Reports**: Automatically generated clinical reports from MRI images.
- **Visual Explanations**: Grad-CAM heatmaps highlighting key regions in the MRI scans.

- Results are saved in the `/results` directory.

## Limitations

- The dataset used for training is artificial and lacks real-world clinical annotations. This could impact the performance and applicability of the model in real clinical environments.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any inquiries or contributions, please reach out to [Vito Nicola Losavio](mailto:v.losavio@studenti.uniba.it).
