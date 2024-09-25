import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
# app.py (in streamlit_app folder)
import sys
import os
import cv2 
# Add the root directory of the project to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Now you can import the ocr function from models.text_extraction_model
from models.segmentation_model import segm
from models.identification_model import identifier
from models.text_extraction_model import extract_text

from utils.postprocessing import postp
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# from models.text_extraction_model import ocr
# from models import text_extraction_model
from transformers import AutoModel, AutoTokenizer

def main():
    st.title("Image Upload App")

    # Name input
    name = st.text_input("Please enter your name:")
    # if not name:
    #     return

    # Image type selection
    image_type = st.selectbox("Select image type:", ["Text Image", "Non-Text Image"])
    if not image_type:
        return

    if image_type == "Non-Text Image" : 
        # Image upload
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.write(f"Hello, {name}! Here's your uploaded {image_type.lower()}:")
            st.image(image, use_column_width=True)

            # Display image details
            st.write("Image Details:")
            st.write("Filename:", uploaded_file.name)
            st.write("Format:", uploaded_file.type)
            st.write("Size:", uploaded_file.size, "bytes")

            # Button to confirm upload
            if st.button("Submit"):
                
                
                image.save(f"..\\data\\input_images\\{uploaded_file.name}")
                with st.spinner("Processing... Hold up tight!"):
                    res = segm(f"..\\data\\input_images\\{uploaded_file.name}",uploaded_file.name)
                st.success("Image submitted successfully!")
                st.write("Result")
                # res_image = Image.fromarray(res)
                img_cv2 = cv2.imread(res)

                # Convert BGR (OpenCV) to RGB (Streamlit)
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

                # Display image
                st.image(img_rgb, caption="Segmented_Image", use_column_width=True)


                # bar = st.progress(0)
                with st.spinner("Processing... Hold up tight!"):
                    res2, detection_results = identifier(f"..\\data\\input_images\\{uploaded_file.name}",uploaded_file.name)
                # Convert BGR (OpenCV) to RGB (Streamlit)
                img_cv2 = cv2.imread(f"..\\data\\input_images\\{uploaded_file.name}")

                x = postp(res2,img_cv2)

                # Convert BGR (OpenCV) to RGB (Streamlit)
                img_rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

                # Display image
                st.image(img_rgb, caption="Detected_Objects", use_column_width=True)
                # bar = st.progress(0)
                st.success("Obects Detected Successfully!")


                # Display detection results
                st.write("Detection Results:")
                st.code(detection_results)

    if image_type=="Text Image":
        # Image upload
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.write(f"Hello, {name}! Here's your uploaded {image_type.lower()}:")
            st.image(image, use_column_width=True)

            # Display image details
            st.write("Image Details:")
            st.write("Filename:", uploaded_file.name)
            st.write("Format:", uploaded_file.type)
            st.write("Size:", uploaded_file.size, "bytes")

            # Button to confirm upload
            if st.button("Submit"):
                
                image.save(f"..\\data\\input_images\\{uploaded_file.name}")
                # bar = st.progress(0)
                with st.spinner("Processing... Hold up tight!"):
                    text = extract_text(f"..\\data\\input_images\\{uploaded_file.name}") 
                # bar.progress(100)
                st.success("Text extracted successfully!")
                st.write("Extracted_Text: " )
                st.code(text)


            

if __name__ == "__main__":
    main()