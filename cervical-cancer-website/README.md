# Cervical Cancer Severity Classification Website

Modern responsive website for the project **Cervical Cancer Severity Classification Using Deep Learning**.

This is a static frontend built with HTML, CSS, and JavaScript. It is customized for the SIPaKMeD Pap smear dataset and includes project sections, dataset cards, methodology workflow, model performance visuals, a dummy image upload prediction interface, team cards, and a contact form.

> Important: The upload prediction feature is a frontend demo only. It generates random dummy results and must not be used for medical diagnosis.

## Folder Structure

```text
cervical-cancer-website/
|-- index.html
|-- README.md
`-- assets/
    |-- css/
    |   `-- style.css
    |-- js/
    |   `-- script.js
    `-- images/
        |-- hero-ai-healthcare.svg
        |-- sample-normal.svg
        |-- sample-parabasal.svg
        |-- sample-mild.svg
        |-- sample-moderate.svg
        |-- sample-severe.svg
        |-- team-placeholder-1.svg
        |-- team-placeholder-2.svg
        `-- team-placeholder-3.svg
```

## Sections Included

- Home page with hero title, subtitle, healthcare AI background, and smooth navigation
- About project section explaining cervical cancer screening and deep learning
- Dataset section with SIPaKMeD details and sample image gallery
- Methodology workflow for preprocessing, feature extraction, CNN model, and severity classification
- Model performance section with progress bars and confusion matrix placeholder
- Upload and prediction section with image preview and dummy JavaScript prediction
- Team section with editable member cards and photo placeholders
- Contact form and footer with social links

## How to Run

Open this file in a browser:

```text
cervical-cancer-website/index.html
```

No build step or backend server is required.

## Dummy Prediction Logic

The JavaScript file listens for an uploaded image, previews it, and randomly selects one of five SIPaKMeD-style classes:

- Superficial-Intermediate
- Parabasal
- Koilocytotic
- Dyskeratotic
- Metaplastic

Each result includes a generated confidence score for demonstration.

To connect a real model later, replace the dummy prediction block in:

```text
assets/js/script.js
```

with an API request to your trained backend model.

## Customization Tips

- Replace placeholder SVGs in `assets/images/` with real approved project visuals.
- Update performance values in `index.html` after model evaluation.
- Replace team names and roles in the Team section.
- Connect the contact form to a backend or form service if needed.

## SIPaKMeD Dataset Note

The full SIPaKMeD dataset is large, approximately 6 GB, so it is not included in this static website folder. Download and store it separately for training, then connect your trained model to the frontend through an API.
