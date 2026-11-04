"""Hostable Streamlit UI for drone-image flood damage assessment."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.config import load_config
from src.constants import CLASS_NAMES, PALETTE
from src.inference import demonstration_mask, predict_mask
from src.model import load_checkpoint
from src.priority import calculate_damage_report
from src.visual import colorize_mask, create_overlay, image_to_png_bytes

DEFAULT_CHECKPOINT = os.getenv("MODEL_PATH", "models/best_model.pth")
DEFAULT_CONFIG = os.getenv("CONFIG_PATH", "config.yaml")


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    return model, checkpoint, device


def class_distribution(mask: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    counts = np.bincount(mask.ravel(), minlength=len(class_names))
    percentages = 100.0 * counts / max(mask.size, 1)
    return pd.DataFrame(
        {
            "Class": class_names,
            "Pixels": counts,
            "Coverage (%)": np.round(percentages, 2),
        }
    )


def priority_badge(level: str) -> str:
    symbol = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(level, "⚪")
    return f"{symbol} {level}"


def load_selected_images(uploaded_files) -> list[tuple[str, Image.Image]]:
    images: list[tuple[str, Image.Image]] = []
    for uploaded in uploaded_files or []:
        try:
            image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            images.append((uploaded.name, image))
        except Exception as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")
    return images


def main() -> None:
    st.set_page_config(
        page_title="Drone Disaster Damage Assessment",
        page_icon="🛰️",
        layout="wide",
    )
    st.title("🛰️ Drone Disaster Damage Assessment")
    st.caption(
        "Semantic segmentation of flooded buildings, roads, water and surrounding infrastructure."
    )

    config = load_config(DEFAULT_CONFIG)
    priority_config = config.get("priority", {})
    checkpoint_path = Path(DEFAULT_CHECKPOINT)
    checkpoint_available = checkpoint_path.exists()

    with st.sidebar:
        st.header("Analysis settings")
        mode_options = ["Trained model"] if checkpoint_available else []
        mode_options.append("UI preview (not AI)")
        mode = st.radio("Inference mode", mode_options, index=0)
        alpha = st.slider("Overlay opacity", 0.10, 0.90, 0.45, 0.05)
        st.divider()
        st.subheader("Optional map location")
        latitude = st.number_input("Latitude", value=0.0, format="%.6f")
        longitude = st.number_input("Longitude", value=0.0, format="%.6f")
        show_map = st.checkbox("Show location on map", value=False)
        st.divider()
        if checkpoint_available:
            st.success(f"Model found: {checkpoint_path.name}")
        else:
            st.warning(
                "No trained checkpoint is included in the ZIP. Train the model and copy "
                "best_model.pth into models/ before using real AI inference."
            )

    uploaded_files = st.file_uploader(
        "Upload one or more drone/aerial images",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
    )

    images = load_selected_images(uploaded_files)
    sample_path = Path("sample_data/demo_scene.png")
    if not images and sample_path.exists():
        st.info("No image uploaded. Showing the included synthetic UI sample.")
        images = [(sample_path.name, Image.open(sample_path).convert("RGB"))]

    if show_map:
        if latitude == 0.0 and longitude == 0.0:
            st.warning("Enter non-zero latitude and longitude values to display a location.")
        else:
            st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=12)

    if not images:
        st.stop()

    model = checkpoint = device = None
    class_names = config["data"].get("class_names", CLASS_NAMES)
    image_size = int(config["data"].get("image_size", 512))
    if mode == "Trained model":
        with st.spinner("Loading segmentation model..."):
            model, checkpoint, device = cached_model(str(checkpoint_path))
        model_config = checkpoint["config"]
        class_names = model_config["data"].get("class_names", CLASS_NAMES)
        image_size = int(model_config["data"].get("image_size", 512))
        st.success(
            f"Using {model_config['model']['architecture']} with "
            f"{model_config['model']['encoder']} encoder."
        )
    else:
        st.warning(
            "Preview mode uses simple color rules only to demonstrate the interface. "
            "Its masks and priority scores are not valid model predictions."
        )

    report_rows: list[dict[str, object]] = []
    for index, (name, image) in enumerate(images):
        st.divider()
        st.subheader(name)
        if mode == "Trained model":
            with st.spinner(f"Analyzing {name}..."):
                mask, original = predict_mask(model, image, image_size, device)
        else:
            mask, original = demonstration_mask(image)

        report = calculate_damage_report(mask, priority_config)
        mask_rgb = colorize_mask(mask)
        overlay = create_overlay(original, mask, alpha)

        col1, col2, col3 = st.columns(3)
        col1.image(original, caption="Original image", use_container_width=True)
        col2.image(mask_rgb, caption="Predicted segmentation", use_container_width=True)
        col3.image(overlay, caption="Damage overlay", use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flooded buildings", f"{report.flooded_building_percent:.2f}%")
        m2.metric("Flooded roads", f"{report.flooded_road_percent:.2f}%")
        m3.metric("Water coverage", f"{report.water_coverage_percent:.2f}%")
        m4.metric("Rescue priority", priority_badge(report.priority_level), f"Score {report.priority_score:.2f}")

        distribution = class_distribution(mask, class_names)
        left, right = st.columns([1.2, 1])
        with left:
            st.write("**Class coverage**")
            st.dataframe(distribution, use_container_width=True, hide_index=True)
        with right:
            st.write("**Coverage chart**")
            st.bar_chart(distribution.set_index("Class")["Coverage (%)"])

        st.download_button(
            "Download overlay PNG",
            data=image_to_png_bytes(overlay),
            file_name=f"{Path(name).stem}_damage_overlay.png",
            mime="image/png",
            key=f"overlay_{index}",
        )

        row = {"image": name, **report.to_dict()}
        if show_map:
            row.update({"latitude": latitude, "longitude": longitude})
        report_rows.append(row)

    st.divider()
    st.subheader("Combined assessment report")
    report_df = pd.DataFrame(report_rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download report CSV",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="disaster_damage_report.csv",
        mime="text/csv",
    )

    with st.expander("Segmentation legend"):
        legend = pd.DataFrame(
            {
                "Class ID": list(range(len(class_names))),
                "Class": class_names,
                "RGB display color": [str(color) for color in PALETTE[: len(class_names)]],
            }
        )
        st.dataframe(legend, use_container_width=True, hide_index=True)

    st.caption(
        "The Low/Medium/High score is a project-defined decision-support rule, not an "
        "official emergency-management standard. Human verification is required."
    )


if __name__ == "__main__":
    main()
