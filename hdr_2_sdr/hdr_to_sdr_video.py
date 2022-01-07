import cv2
import numpy as np
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from hdr_to_sdr import SdrConversion

from tqdm import tqdm

if __name__ == '__main__':
    SDRConvention = SdrConversion()
    # file_name = 'journey_04'
    # rootDir = f'C:\\Users\\liumm\\Videos\\SZ-video-img\\{file_name}'
    # writedir = f'C:\\Users\\liumm\\Videos\\SZ-video-img-sdr-v2\\{file_name}_sdr_v2'

    imgrootdir = f'C:\\Users\\liumm\\Videos\\delta-video-img\\'
    imgwritedir = f'C:\\Users\\liumm\\Videos\\delta-video-img-sdr-v4\\'
    if not os.path.exists(imgwritedir):os.makedirs(imgwritedir)

    for img_dir_name in tqdm(os.listdir(imgrootdir)):
        img_dir_path = os.path.join(imgrootdir, img_dir_name)
        if os.path.isdir(img_dir_path):
            rootDir = img_dir_path
            writedir = os.path.join(imgwritedir, img_dir_name)
            if not os.path.exists(writedir):os.makedirs(writedir)

            #----------generate SDR(with gamut-conv) imgs-------------------------------------------
            for file_name in (os.listdir(rootDir)): 
                image_path = os.path.join(rootDir, file_name)
                image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 65535.
                image_rgb = np.float32(image_bgr[..., ::-1].copy())
                # image_rgb =  np.from_numpy(image_rgb)
                image_rgb_hdr_nolinear = SDRConvention.hdr_to_sdr(image_rgb)
                # image_rgb_hdr_nolinear = image_rgb_hdr_nolinear.cpu().detach().numpy()
                image_bgr_hdr_nolinear = image_rgb_hdr_nolinear[..., ::-1]
                # cv2.imwrite(f'{writedir}\\{file_name}', np.uint8(image_bgr_hdr_nolinear * 255.))
                cv2.imwrite(f'{writedir}\\{file_name}', (image_bgr_hdr_nolinear * 255.))
            #---------------------------------------------------------------------------------------        
        else:
            print(f'{img_dir_path} -- is not a folder !')
    


