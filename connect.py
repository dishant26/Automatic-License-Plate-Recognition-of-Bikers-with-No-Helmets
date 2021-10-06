import os
import io
import re
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import cv2
import csv
import shutil

my_dir = os.getcwd()

if os.path.exists(f'{my_dir}\\helpers'):
    shutil.rmtree(f'{my_dir}\\helpers')
os.mkdir(f'{my_dir}\\helpers\\')

if os.path.exists(f'{my_dir}\\helpers\\bike_crop_imgs'):
    shutil.rmtree(f'{my_dir}\\helpers\\bike_crop_imgs')
os.mkdir(f'{my_dir}\\helpers\\bike_crop_imgs\\')

if os.path.exists(f'{my_dir}\\final_output'):
    shutil.rmtree(f'{my_dir}\\final_output')
os.mkdir(f'{my_dir}\\final_output\\')

if os.path.exists(f'{my_dir}\\helpers\\number_plates'):
    shutil.rmtree(f'{my_dir}\\helpers\\number_plates')
os.mkdir(f'{my_dir}\\helpers\\number_plates\\')

if os.path.exists(f'{my_dir}\\helpers\\bike_csv'):
    shutil.rmtree(f'{my_dir}\\helpers\\bike_csv')

if os.path.exists(f'{my_dir}helpers\\bike_data_csv'):
    shutil.rmtree(f'{my_dir}helpers\\bike_data_csv')



image_path = 'C:\\Users\\disha\\Desktop\\Internship\\Datasets\\stage1_data\\images\\val\\193.png'

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
    os.system(f"python detect.py --weights runs/train/exp/weights/last.pt --img 928 --conf 0.50 --source {img_path}")

    if os.path.exists(f'{my_dir}\\helpers\\bike_data_csv\\out.csv'):
        this_df = pd.read_csv(f'{my_dir}\\helpers\\bike_data_csv\\out.csv')

        this_df['image_path'] = img_path
        final_df = final_df.append(this_df, ignore_index=True)

        os.remove(f'{my_dir}\\helpers\\bike_data_csv\\out.csv')

final_df.to_csv(f'{my_dir}\\final_output\\output.csv')