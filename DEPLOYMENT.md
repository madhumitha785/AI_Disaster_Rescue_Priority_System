# Deployment

## Before deployment

Train a model and copy the best checkpoint:

```bash
mkdir -p models
cp outputs/unet_mobilenetv2/best_model.pth models/best_model.pth
```

On Windows, copy the file manually into `models\best_model.pth`.

The app can start without the checkpoint, but it will only show the clearly labelled UI preview mode. Real predictions require the trained checkpoint.

## Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Store a large checkpoint using Git LFS when required.
3. In Streamlit Community Cloud, select the repository.
4. Set the entrypoint to `app.py`.
5. Deploy. The included `requirements.txt` is used for dependencies.

Optional environment variable:

```text
MODEL_PATH=models/best_model.pth
```

## Hugging Face Spaces with Docker

1. Create a new Space using the Docker SDK.
2. Upload this project, including `Dockerfile`.
3. Add `models/best_model.pth` or download it securely during container startup.
4. The Dockerfile exposes Streamlit on port 8501.

## Local Docker

```bash
docker build -t disaster-assessment .
docker run --rm -p 8501:8501 disaster-assessment
```

Open `http://localhost:8501`.

## Important

Do not deploy the training dataset with the app. Only deploy source code, configuration, class names, and the trained inference checkpoint.
