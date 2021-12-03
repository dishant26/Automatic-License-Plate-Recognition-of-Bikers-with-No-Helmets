<div>

 ## Automatic License Plate-Recognition of Bikers with No Helmets

Motorcycles have always been the primary mode of transport in developing countries. In recent years, there has been a rise in motorcycle accidents. One of the major reasons for fatalities in accidents is the motorcyclist not wearing a protective helmet. The most prevalent method for ensuring that motorcyclists wear helmet is traffic police manually monitoring motorcyclists at road junctions or through CCTV footage and penalizing those without helmet.

## Tech Stack:
[![](https://img.shields.io/badge/Made_with-Python-red?style=for-the-badge&logo=python)](https://www.python.org/)
[![](https://img.shields.io/badge/Made_with-TensorFlow-red?style=for-the-badge&logo=TensorFlow)](https://www.tensorflow.org/)
[![](https://img.shields.io/badge/Made_with-YOLOv5-red?style=for-the-badge&logo=)](https://github.com/ultralytics/yolov5)
[![](https://img.shields.io/badge/Made_with-Keras-red?style=for-the-badge&logo=Keras)](https://www.keras.io/)

## Sample Demo:

**Image from the CCTV Camera:**

 <img src="Assets/49_train.png" width=640px height=360px/>

**Vehicle Detected by the First Stage YOLO:**

 <img src="Assets/49.png" width=640px height=360px/>
 
**Number Plate and Helmet/No Helmet Detected by Second Stage YOLO:**

 <img src="Assets/img_0.png" width=214x height=414px/>

**Number Plate Cropped Automatically:**
 
 <img src="helpers/number_plates/plate_0.png" width=141x height=65px/>
 
 
**The Number Plate is Detected using OCR using the Google Cloud Vision/ Tessaract:**
 
 **Detected : MH 12 DZ 2102**
 
</div>
