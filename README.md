# License Plate Recognition Using Image Processing and OCR


This project aims to automatically recognize license plates from car images using image processing techniques and Optical Character Recognition (OCR). The LPR system efficiently identifies and extracts license plate information, making it useful for various applications such as parking management, traffic surveillance, and security systems.

## How It Works

1. **Image Preprocessing:** We use OpenCV and NumPy for image resizing and Gaussian smoothing to enhance the quality of the input images.

2. **Edge Detection:** Canny edge detection is applied to identify edges within the images, which is crucial for locating the license plate regions.

3. **Contour Detection:** Contours of the car images are detected using the Canny edges, and the top 30 contours are selected based on their area.

4. **License Plate Extraction:** Our algorithm attempts to find the license plate contour by approximating its vertices. When a four-sided contour is identified, the license plate region is cropped and saved as a separate image.

5. **OCR and Verification:** The cropped license plate images are processed by Tesseract OCR to extract the alphanumeric characters. The recognized text is then compared with the ground truth data (stored in the 'ground_truth.csv' file) to validate the license plate.

