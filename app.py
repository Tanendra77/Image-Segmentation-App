import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Segmentation Functions
def thresholding_segmentation(image):
    """1. Thresholding Segmentation"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

def kmeans_segmentation(image, k=3):
    """2. K-means Clustering"""
    reshaped = image.reshape((-1, 3))
    reshaped = np.float32(reshaped)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    return segmented.reshape(image.shape)

def canny_edge_detection(image):
    """3. Canny Edge Detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def watershed_segmentation(image):
    """4. Watershed Algorithm"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    background = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, foreground = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    foreground = np.uint8(foreground)
    unknown = cv2.subtract(background, foreground)
    _, markers = cv2.connectedComponents(foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    result = image.copy()
    result[markers == -1] = [255, 0, 0]
    return result

# Streamlit App
st.title("Image Segmentation App")

# Image Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to OpenCV image
    pil_image = Image.open(uploaded_file).convert("RGB")
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    # Display original image
    st.image(pil_image, caption="Original Image", use_column_width=True)

    # Apply Segmentation
    st.header("Segmentation Results")
    
    # Thresholding
    thresholded = thresholding_segmentation(cv_image)
    st.image(thresholded, caption="Thresholding Segmentation", use_column_width=True, channels="GRAY")
    
    # K-Means Clustering
    kmeans_result = kmeans_segmentation(cv_image, k=3)
    st.image(kmeans_result, caption="K-Means Clustering", use_column_width=True)
    
    # Canny Edge Detection
    canny_edges = canny_edge_detection(cv_image)
    st.image(canny_edges, caption="Canny Edge Detection", use_column_width=True, channels="GRAY")
    
    # Watershed Segmentation
    watershed_result = watershed_segmentation(cv_image)
    st.image(watershed_result, caption="Watershed Segmentation", use_column_width=True)


   
