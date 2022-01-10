import cv2
import numpy as np
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from hdr_2_sdr.hdr_to_sdr import SdrConversion

if __name__ == '__main__':
    SDRConvention = SdrConversion(cuda_available=False)
    # image_path = r"../example-data/hdr.png"
    image_path = r"D:\\Code\\VideoHDR-GamutConv\\example-data\\0000050.png"
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 65535.
    image_rgb = np.float32(image_bgr[..., ::-1].copy())
    image_rgb = torch.from_numpy(image_rgb)
    result_rgb_image = SDRConvention.hdr_to_sdr(image_rgb)
    result_rgb_image = result_rgb_image.cpu().numpy()
    # cv2.imwrite(r"D:\\Code\\VideoHDR-mm\\example-data\\0000002_sdr_no.png", np.uint8(result_rgb_image[..., ::-1] * 255.))
    cv2.imwrite(r"D:\\Code\\VideoHDR-GamutConv\\example-data\\0000050_sdr_v3_vifi_tst.png", (result_rgb_image[..., ::-1] * 255.))
