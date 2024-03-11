import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

# Function to send image to Azure Custom Vision API and get predictions
def send_image_to_custom_vision_api(image_bytes):
    url = "https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/bc29247c-c3d7-42de-9211-d175f58fae3a/detect/iterations/Iteration5/image"
    headers = {
        "Prediction-Key": "2eae9430f9eb425ea070d783ad12f0ee",
        "Content-Type": "application/octet-stream",
    }
    response = requests.post(url, headers=headers, data=image_bytes)
    return response.json()

# Function to draw the most accurate bounding box on the image
def draw_most_accurate_bounding_box(image, predictions):
    if predictions['predictions']:
        most_accurate = max(predictions['predictions'], key=lambda x: x['probability'])
        draw = ImageDraw.Draw(image)
        bbox = most_accurate['boundingBox']
        left = bbox['left'] * image.width
        top = bbox['top'] * image.height
        width = bbox['width'] * image.width
        height = bbox['height'] * image.height

        draw.rectangle([left, top, left + width, top + height], outline="#2ECC71", width=3)  # Green outline
        # Tag and probability text is now moved to be displayed below the image.

        return most_accurate['tagName'], most_accurate['probability']
    return "", 0.0  # Return empty tag and zero probability if no predictions

def main():
    st.title("Image Prediction with Azure Custom Vision")
    st.sidebar.header("Configuration")
    st.sidebar.info("This app uses Azure Custom Vision to predict objects in images. Upload an image, and it will display the most accurate prediction along with its bounding box.")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        with st.spinner('Analyzing the image...'):
            image = Image.open(uploaded_file).convert("RGB")

            # Convert the PIL Image to bytes for the API request
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()

            result = send_image_to_custom_vision_api(image_bytes)
            tag_name, probability = draw_most_accurate_bounding_box(image, result)

        st.success('Analysis complete.')

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.header("Original Image")
            st.image(uploaded_file, use_column_width=True)

        with col2:
            st.header("Prediction Result")
            st.image(image, use_column_width=True)

        if tag_name:
            st.subheader("Most Accurate Prediction")
            st.markdown(f"**Tag:** `{tag_name}`")
            st.markdown(f"**Probability:** `{probability:.2f}`")
            st.progress(probability)

        if result['predictions']:
            st.subheader("Details of Most Accurate Prediction")
            st.json(max(result['predictions'], key=lambda x: x['probability']))
        else:
            st.warning("No predictions made. Please try a different image.")

if __name__ == "__main__":
    main()
