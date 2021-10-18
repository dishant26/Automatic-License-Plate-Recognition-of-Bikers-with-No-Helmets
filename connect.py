import os
import io
import re
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import cv2
import csv
import shutil
import easyocr

my_dir = os.getcwd()

if os.path.exists(f'{my_dir}\\helpers'):
    shutil.rmtree(f'{my_dir}\\helpers')
os.mkdir(f'{my_dir}\\helpers\\')

if os.path.exists(f'{my_dir}\\helpers\\bike_crop_imgs'):
    shutil.rmtree(f'{my_dir}\\helpers\\bike_crop_imgs')
os.mkdir(f'{my_dir}\\helpers\\bike_crop_imgs\\')

if os.path.exists(f'{my_dir}\\helpers\\number_plate_dimension'):
    shutil.rmtree(f'{my_dir}\\helpers\\number_plate_dimension')
os.mkdir(f'{my_dir}\\helpers\\number_plate_dimension\\')

if os.path.exists(f'{my_dir}\\helpers\\number_plates'):
    shutil.rmtree(f'{my_dir}\\helpers\\number_plates')
os.mkdir(f'{my_dir}\\helpers\\number_plates\\')

if os.path.exists(f'{my_dir}\\helpers\\bike_csv'):
    shutil.rmtree(f'{my_dir}\\helpers\\bike_csv')

if os.path.exists(f'{my_dir}helpers\\bike_data_csv'):
    shutil.rmtree(f'{my_dir}helpers\\bike_data_csv')



image_path = 'C:\\Users\\disha\\Desktop\\Internship\\Datasets\\stage1_data\\images\\train\\49.png'

os.system(f"python detect.py --weights runs/train/exp5/weights/last.pt --img 928 --conf 0.50 --source {image_path}")

bike_dim = pd.read_csv(f"{my_dir}\\helpers\\bike_csv\\bike_dimensions.csv")
bike_dim['x1'] = bike_dim['x1'].astype(int)
bike_dim['x2'] = bike_dim['x2'].astype(int)
bike_dim['y1'] = bike_dim['y1'].astype(int)
bike_dim['y2'] = bike_dim['y2'].astype(int)

bike_crop_imgs_folder = f'{my_dir}\\helpers\\bike_crop_imgs\\'

image = cv2.imread(image_path)
for i in range(len(bike_dim)):
    crp_img = image[bike_dim['y1'][i]:bike_dim['y2'][i], bike_dim['x1'][i]:bike_dim['x2'][i]]
    cv2.imwrite(f'{bike_crop_imgs_folder}img_{i}.png', crp_img)

bike_imgs_list = os.listdir(bike_crop_imgs_folder)

os.chdir(f"{my_dir}\\Stage2\\")

final_df = pd.DataFrame()

for i in range(len(bike_imgs_list)):
    img_path = bike_crop_imgs_folder + bike_imgs_list[i]
    os.system(f"python detect.py --weights runs/train/exp/weights/last.pt --img 928 --conf 0.30 --source {img_path}")

    if os.path.exists(f'{my_dir}\\helpers\\bike_data_csv\\out.csv'):
        this_df = pd.read_csv(f'{my_dir}\\helpers\\bike_data_csv\\out.csv')

        this_df['image_path'] = img_path
        final_df = final_df.append(this_df, ignore_index=True)

        os.remove(f'{my_dir}\\helpers\\bike_data_csv\\out.csv')

final_df.to_csv(f'{my_dir}\\helpers\\number_plate_dimension\\output.csv', index=False)

try:
    plate_dim = pd.read_csv(f"{my_dir}\\helpers\\number_plate_dimension\\output.csv")
except:
    pass


plate_dim['x1'] = plate_dim['x1'].astype(int)
plate_dim['x2'] = plate_dim['x2'].astype(int)
plate_dim['y1'] = plate_dim['y1'].astype(int)
plate_dim['y2'] = plate_dim['y2'].astype(int)

for i in range(len(plate_dim)):
    temp = plate_dim['label'][i].split(" ")
    if temp[0] == "number_plate":
        path = r'{}'.format(plate_dim['image_path'][i])
        im = cv2.imread(path)
        number_plat_crp = im[plate_dim['y1'][i]:plate_dim['y2'][i], plate_dim['x1'][i]:plate_dim['x2'][i]]
        cv2.imwrite(f'{my_dir}\\helpers\\number_plates\\plate_{i}.png', number_plat_crp)

reader = easyocr.Reader(['en'], gpu=False)

plate_folder_path = f"{my_dir}\\helpers\\number_plates"
for i in os.listdir(plate_folder_path):
    plate_path = os.path.join(plate_folder_path, i)
    result = reader.readtext(plate_path, detail=0)
    plate_number = " ".join(result)
    print()
    print(plate_number)


