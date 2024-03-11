import cv2
import requests
import time
import io
from PIL import Image

def send_frame_to_custom_vision_api(image_bytes):
    url = "https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/bc29247c-c3d7-42de-9211-d175f58fae3a/detect/iterations/Iteration5/image"
    headers = {
        "Prediction-Key": "2eae9430f9eb425ea070d783ad12f0ee",
        "Content-Type": "application/octet-stream",
    }
    response = requests.post(url, headers=headers, data=image_bytes)
    return response.json()

# Initialize video capture from a file or camera
cap = cv2.VideoCapture(0)

# Interval settings
capture_interval = 1  # Capture frame every 1 second
print_interval = 60  # Print results every 60 seconds

last_capture_time = time.time()
last_print_time = time.time()

# Temporary storage for accumulated results
accumulated_results = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Capture and process frame at the defined interval
        if current_time - last_capture_time >= capture_interval:
            last_capture_time = current_time

            # Convert the frame to a format suitable for Azure Custom Vision API
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()

            # Send frame to Custom Vision API and accumulate results
            response = send_frame_to_custom_vision_api(image_bytes)
            accumulated_results.append(response)

        # Display the resulting frame (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Print accumulated results at the defined interval
        if current_time - last_print_time >= print_interval:
            last_print_time = current_time
            print(f"Results accumulated over the last {print_interval} seconds:")
            for result in accumulated_results:
                # Process and print results
                # Assuming 'result' structure includes 'predictions' with 'tagName' and 'probability'
                for prediction in result['predictions']:
                    print(f"Detected: {prediction['tagName']} with probability {prediction['probability']:.2f}")
            
            # Clear the accumulated results after printing
            accumulated_results = []

finally:
    cap.release()
    cv2.destroyAllWindows()
