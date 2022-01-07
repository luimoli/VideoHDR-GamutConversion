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

    imgrootdir = f'C:\\Users\\liumm\\Videos\\max-delta-video-img\\'
    # imgwritedir = f'C:\\Users\\liumm\\Videos\\delta-video-img-sdr-v0\\'
    # if not os.path.exists(imgwritedir):os.makedirs(imgwritedir)
    post = ['v0', 'v1', 'v2', 'v3']
    imgwritedir_list = [f'C:\\Users\\liumm\\Videos\\max-delta-video-img-sdr-{x}\\' for x in post]

    pd_list = [] # # to save delta_E list results row by row
    pd_list_with_v0 = []
    for img_dir_name in tqdm(os.listdir(imgrootdir)):
        img_dir_path = os.path.join(imgrootdir, img_dir_name)
        if os.path.isdir(img_dir_path):
            rootDir = img_dir_path
            # writedir = os.path.join(imgwritedir, img_dir_name)
            # if not os.path.exists(writedir):os.makedirs(writedir)

            # #---------calculate delta-E of SDR imgs------------------------------------------------
            img_dir_name_only = img_dir_name.split('.')[0]
            for file_name in os.listdir(rootDir): 
                image_path = os.path.join(rootDir, file_name)
                image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 65535.
                image_rgb = np.float32(image_bgr[..., ::-1].copy())
                res_list, deltaE_list, deltaE_list_with_v0 = SDRConvention.hdr_to_sdr(image_rgb)

                #-------save img-------------------------------
                writedir = [os.path.join(x, img_dir_name) for x in imgwritedir_list]
                for i in range(len(res_list)):
                    if not os.path.exists(writedir[i]): os.makedirs(writedir[i])
                    res = res_list[i][..., ::-1]
                    cv2.imwrite(f'{writedir[i]}\\{file_name}', (res * 255.))
                #----------------------------------------------

                deltaE_list.append(img_dir_name_only)
                deltaE_list_with_v0.append(img_dir_name_only)
                print(deltaE_list)
                print(deltaE_list_with_v0)
            pd_list.append(deltaE_list)
            pd_list_with_v0.append(deltaE_list_with_v0)            
        else:
            print(f'{img_dir_path} -- is not a folder !')
    
    df = pd.DataFrame(pd_list, columns =['RGB_clipping','TWP + xyY','Closest + xyY', 'Closest + uvY', 'img_name'])
    df.to_csv(f'D:\\Code\\VideoHDR-mm\\max_deltaE_SDR_cmp_with_HDR.csv', index=False)

    df_with_v0 = pd.DataFrame(pd_list_with_v0, columns =['RGB_clipping','TWP + xyY','Closest + xyY', 'Closest + uvY', 'img_name'])
    df_with_v0.to_csv(f'D:\\Code\\VideoHDR-mm\\max_deltaE_SDR_cmp_with_RGB_clipping.csv', index=False)

    np.save(f'D:\\Code\\VideoHDR-mm\\max_deltaE_SDR_cmp_with_HDR.npy', pd_list)
    np.save(f'D:\\Code\\VideoHDR-mm\\max_deltaE_SDR_cmp_with_RGB_clipping.npy', pd_list_with_v0)


