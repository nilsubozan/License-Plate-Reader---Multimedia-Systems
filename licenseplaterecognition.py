#imutils is imported to resize the image.
import csv
import cv2
import numpy as np
import imutils
import pytesseract 
import matplotlib.pyplot as plt
import os

folder_path = '/Users/Lenovo/Desktop/itu-3.sinif-bahar-donemi/ISE309-multimedya/proje/proje_yuklenecek/cars'

#All the files in the folder are listed
image_files = os.listdir(folder_path)
image_files = sorted(image_files)

ground_truth_file = '/Users/Lenovo/Desktop/itu-3.sinif-bahar-donemi/ISE309-multimedya/proje/proje_yuklenecek/ground_truth.csv'

rows=[]
with open(ground_truth_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        rows.append(row[1])
    
# Loop over each image file

counter=0
for image_file in image_files:

    image_path = os.path.join(folder_path, image_file)
   
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
    else:

        img=imutils.resize(img, width=500)
        grayim=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #IMAGE DISPLAY
        cv2.imshow('GRAYSCALE IMAGE', grayim)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        GaussianKernel= np.array([[ 1, 4, 7,  4,  1 ],
                                [ 4, 16, 26, 16,  4 ],
                                [ 7, 26, 41, 26,  7 ],
                                [ 4, 16, 26, 16,  4 ],
                                [ 1,  4,  7,  4,  1 ]])

        def normalize_kernel(kernel):
            return kernel / np.sum(kernel)

        normalized_gaussian_kernel= normalize_kernel(GaussianKernel)

        def padWith(kernel_size):
            pw = kernel_size // 2
            return pw

        def zeroPadding(img, pw):
            height = img.shape[0]
            width = img.shape[1]
            padded_height = height + 2 * pw
            padded_width = width + 2 * pw
            padded_image = np.zeros((padded_height, padded_width), dtype=img.dtype)

            padded_image[pw:height+pw, pw:width+pw] = img

            return padded_image

        def convolution(padded_image, img, kernel):
            output = np.zeros_like(img)
            for y in range(0, img.shape[0]):
                for x in range(0, img.shape[1]):
                    output[y, x] = np.sum(kernel * padded_image[y:y + kernel.shape[0], x:x + kernel.shape[1]])
            return output

        Pwith = padWith(normalized_gaussian_kernel.shape[0]) #Pad with filter size
        paddedImg = zeroPadding(grayim, Pwith) #Padded Image
        img_convolution = convolution(paddedImg, grayim, normalized_gaussian_kernel )


        grayim = img_convolution


        #IMAGE DISPLAY
        cv2.imshow('smooth image', grayim)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        edge= cv2.Canny(img,170,200)

        #IMAGE DISPLAY
        cv2.imshow('Canny edge detection', edge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        cnts, new=cv2.findContours(edge.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        img1=img.copy()

        cv2.drawContours(img1,cnts,-1,(0,255,0),3)

        #IMAGE DISPLAY
        cv2.imshow('Contours', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cnts=sorted(cnts, key=cv2.contourArea,reverse=True)[:30]
        numPlateCount=None    

        img2=img.copy()
        cv2.drawContours(img2,cnts,-1,(0,255,0),3)

        #IMAGE DISPLAY
        cv2.imshow('Top 30 contours', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        count=0
        name=1

        for i in cnts:
            perimeter=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*perimeter,True)
            if(len(approx)==4):
                numPlateCount=approx
                x,y,w,h=cv2.boundingRect(i)
                crp_img=img[y:y+h,x:x+w]

                cv2.imwrite(str(name)+ '.png',crp_img)
                name+=1

                #cropped image will be displayed later
                cropped_image = crp_img
            

                break
        
        try:
            cv2.drawContours(img, [numPlateCount], -1, (0, 255, 0), 3)
        except cv2.error as e:
            print("Error occurred while drawing contours:", e)
        
        
        #IMAGE DISPLAY
        cv2.imshow('Final Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the cropped image
        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Read the number plate
        text = pytesseract.image_to_string(cropped_image, config='--psm 7')
        print("Detected license plate Number is:",text)
        if(text.strip().lower() == rows[counter].strip().lower()):
            print('We can let the car in.')
        else:
            print('This license plate does not match the license plates in our system!')
 
        counter=counter+1

    
    

       


