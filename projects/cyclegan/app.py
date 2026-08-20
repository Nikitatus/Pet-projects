"""Streamlit application for photo-to-painting CycleGAN inference."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as transform_functional

from model import Generator


APP_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = APP_DIR / "weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIDE = 768

WEIGHT_PATHS = {
    "Van Gogh": WEIGHTS_DIR / "van_gogh_generator.pt",
    "Monet": WEIGHTS_DIR / "monet_generator.pt",
}


@st.cache_resource(show_spinner=False)
def load_generator(style: str) -> Generator:
    """Load a style generator once and reuse it across Streamlit reruns."""
    if style not in WEIGHT_PATHS:
        raise ValueError(f"Unknown style: {style}")

    weight_path = WEIGHT_PATHS[style]
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"The {style} weights are not installed. Expected: "
            f"weights/{weight_path.name}"
        )

    generator = Generator().to(DEVICE)
    state_dict = torch.load(
        weight_path,
        map_location=DEVICE,
        weights_only=True,
    )
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator


def prepare_image(image: Image.Image) -> torch.Tensor:
    """Convert an uploaded image to the normalized model input tensor."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS,
    )

    width, height = image.size
    # Eight pixels is the safe minimum for reflection padding after two
    # stride-two downsampling layers.
    width = max(8, (width // 4) * 4)
    height = max(8, (height // 4) * 4)
    image = image.resize((width, height), Image.Resampling.LANCZOS)

    tensor = transform_functional.to_tensor(image)
    tensor = transform_functional.normalize(
        tensor,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    return tensor.unsqueeze(0).to(DEVICE)


def tensor_to_image(output: torch.Tensor) -> Image.Image:
    output = output.squeeze(0).clamp(-1, 1)
    output = (output + 1) / 2
    return transform_functional.to_pil_image(output.cpu())


@torch.inference_mode()
def stylize_both(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Generate both styles from one preprocessed input image."""
    input_tensor = prepare_image(image)
    van_gogh_output = load_generator("Van Gogh")(input_tensor)
    monet_output = load_generator("Monet")(input_tensor)
    return tensor_to_image(van_gogh_output), tensor_to_image(monet_output)


def image_as_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(
        page_title="CycleGAN Art Style Transfer",
        page_icon="🎨",
        layout="wide",
    )

    st.title("CycleGAN Art Style Transfer")
    st.write(
        "Upload one photograph to generate **Van Gogh** and **Monet** "
        "versions together. Large images are resized to a maximum side of "
        f"{MAX_IMAGE_SIDE} pixels."
    )

    uploaded_file = st.file_uploader(
        "Upload a photograph",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        original = Image.open(uploaded_file)
        original.load()
        original = ImageOps.exif_transpose(original).convert("RGB")
    except (OSError, ValueError) as error:
        st.error(f"The uploaded file could not be read as an image: {error}")
        return

    st.image(original, caption="Original photograph", width=500)

    upload_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state["upload_signature"] = upload_signature
        st.session_state.pop("result_images", None)

    generate_clicked = st.button(
        "Generate both styles",
        type="primary",
        width="stretch",
    )

    if generate_clicked:
        try:
            with st.spinner("Generating Van Gogh and Monet versions..."):
                van_gogh_image, monet_image = stylize_both(original)
            st.session_state["result_images"] = (
                image_as_png(van_gogh_image),
                image_as_png(monet_image),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.error(f"Generation failed: {error}")
            st.session_state.pop("result_images", None)
            return

    result_images = st.session_state.get("result_images")
    if result_images is None:
        return

    van_gogh_png, monet_png = result_images

    van_gogh_column, monet_column = st.columns(2)

    with van_gogh_column:
        st.subheader("Van Gogh style")
        st.image(van_gogh_png, width="stretch")
        st.download_button(
            "Download Van Gogh image",
            data=van_gogh_png,
            file_name="van_gogh_style.png",
            mime="image/png",
            width="stretch",
        )

    with monet_column:
        st.subheader("Monet style")
        st.image(monet_png, width="stretch")
        st.download_button(
            "Download Monet image",
            data=monet_png,
            file_name="monet_style.png",
            mime="image/png",
            width="stretch",
        )


if __name__ == "__main__":
    main()
