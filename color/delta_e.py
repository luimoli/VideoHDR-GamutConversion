import cv2
import numpy as np
import matplotlib.pyplot as plt
# from color import color_model
import color_model
import math
import colour


class DeltaE_cmp():
    def __init__(self) -> None:
        self.RGB709_to_XYZ_matrix =  np.array([[0.412391, 0.357584, 0.180481],
                                                  [0.212639, 0.715169, 0.072192],
                                                  [0.019331, 0.119195, 0.950532]], dtype= np.float32)
        
        self.XYZ_to_RGB709_matrix = np.array([[ 3.2409663 , -1.5373788 , -0.49861172],
                                                    [-0.96924204,  1.8759652 ,  0.04155577],
                                                    [ 0.05562956, -0.20397693,  1.0569717 ]], dtype=np.float32)

        #------------------------------------------------------------------------------------------

        self.RGB2020_to_XYZ_matrix =  np.array([[0.636958, 0.144617, 0.168881],
                                                   [0.262700, 0.677998, 0.059302],
                                                   [0.000000, 0.028073, 1.060985]], dtype= np.float32)

        self.XYZ_to_RGB2020_matrix =  np.array([[1.716651, -0.355671, -0.253366],
                                                   [-0.666684, 1.616481, 0.015769],
                                                   [0.017640, -0.042771, 0.942103]], dtype= np.float32)
                                                   
    def delta_E_CIE2000(self, Lab_1, Lab_2, textiles=False):

        L_1, a_1, b_1 = [Lab_1[..., x] for x in range(Lab_1.shape[-1])]
        L_2, a_2, b_2 = [Lab_2[..., x] for x in range(Lab_2.shape[-1])]

        k_L = 2 if textiles else 1
        k_C = 1
        k_H = 1

        l_bar_prime = 0.5 * (L_1 + L_2)

        c_1 = np.hypot(a_1, b_1)
        c_2 = np.hypot(a_2, b_2)

        c_bar = 0.5 * (c_1 + c_2)
        c_bar7 = c_bar ** 7

        g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

        a_1_prime = a_1 * (1 + g)
        a_2_prime = a_2 * (1 + g)
        c_1_prime = np.hypot(a_1_prime, b_1)
        c_2_prime = np.hypot(a_2_prime, b_2)
        c_bar_prime = 0.5 * (c_1_prime + c_2_prime)

        h_1_prime = np.degrees(np.arctan2(b_1, a_1_prime)) % 360
        h_2_prime = np.degrees(np.arctan2(b_2, a_2_prime)) % 360

        h_bar_prime = np.where(
            np.fabs(h_1_prime - h_2_prime) <= 180,
            0.5 * (h_1_prime + h_2_prime),
            (0.5 * (h_1_prime + h_2_prime + 360)),
        )

        t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
            0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
            0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
            0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

        h = h_2_prime - h_1_prime
        delta_h_prime = np.where(h_2_prime <= h_1_prime, h - 360, h + 360)
        delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

        delta_L_prime = L_2 - L_1
        delta_C_prime = c_2_prime - c_1_prime
        delta_H_prime = (2 * np.sqrt(c_1_prime * c_2_prime) * np.sin(
            np.deg2rad(0.5 * delta_h_prime)))

        s_L = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
                np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
        s_C = 1 + 0.045 * c_bar_prime
        s_H = 1 + 0.015 * c_bar_prime * t

        delta_theta = (
            30 * np.exp(-((h_bar_prime - 275) / 25) * ((h_bar_prime - 275) / 25)))

        c_bar_prime7 = c_bar_prime ** 7

        r_C = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
        r_T = -2 * r_C * np.sin(np.deg2rad(2 * delta_theta))

        d_E = np.sqrt((delta_L_prime / (k_L * s_L)) ** 2 +
                    (delta_C_prime / (k_C * s_C)) ** 2 +
                    (delta_H_prime / (k_H * s_H)) ** 2 +
                    (delta_C_prime / (k_C * s_C)) * (delta_H_prime /
                                                    (k_H * s_H)) * r_T)

        return d_E
    
    def delta_E_metric(self, ori_lab, cmp_lab_list, metric='2000'):
        '''
        ori_lab: img in lab color space
        cmp_lab_list: img in lab color space <list>
        metric: '1976' | '1994' | '2000' | 'cmc'
        '''
        if metric == '1976': delta = [colour.difference.delta_E_CIE1976(ori_lab, x) for x in cmp_lab_list]
        elif metric == '1994': delta = [colour.difference.delta_E_CIE1994(ori_lab, x) for x in cmp_lab_list]
        # elif metric == '2000': delta = [colour.difference.delta_E_CIE2000(ori_lab, x) for x in cmp_lab_list]
        elif metric == '2000': delta = [self.delta_E_CIE2000(ori_lab, x) for x in cmp_lab_list]
        elif metric == 'cmc': delta = [colour.difference.delta_E_CMC(ori_lab, x) for x in cmp_lab_list]
        else: print('Wrong delta_E metric choice!')
        delta_n = [np.where(np.isnan(delta),  np.zeros_like(delta), delta) for delta in delta]
        # img_mean_deltaE = [format(np.mean(delta_n), '.4f') for delta_n in delta_n]

        # return img_mean_deltaE
        return delta_n


    def cmp(self, ori_img, cmp_img_list, ori_type='709',cmp_type='709'):
        '''
        ori_img: RGB [0,1]
        cmp_img_list: RGB [0,1] <list>
        ori_type/cmp_type : '2020' or '709'
        '''
        if ori_type == '2020':
            ori_xyz =  np.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, ori_img)  
        elif ori_type == '709':
            ori_xyz =  np.einsum('ic,hwc->hwi', self.RGB709_to_XYZ_matrix, ori_img)
        else:
            print('wrong ori_img_type when calculate delta_E!!!')
        ori_lab =  colour.XYZ_to_Lab(ori_xyz)
    
        if cmp_type == '2020':
            cmp_xyz =  [np.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, x) for x in cmp_img_list]
        elif cmp_type == '709':
            cmp_xyz =  [np.einsum('ic,hwc->hwi', self.RGB709_to_XYZ_matrix, x) for x in cmp_img_list]
        else:
            print('wrong cmp_img_type when calculate delta_E!!!')
        cmp_lab_list = [colour.XYZ_to_Lab(x) for x in cmp_xyz]

        res = self.delta_E_metric(ori_lab, cmp_lab_list, metric='2000')

        return res

if __name__ == '__main__':
    DeltaE_cmp = DeltaE_cmp()

    # name = 'green'
    # r_channel = np.uint8(np.ones((540,960,1)) * 70.0)
    # g_channel = np.uint8(np.ones((540,960,1)) * 230.0)
    # b_channel = np.uint8(np.ones((540,960,1)) * 70.0)
    # img_rgb = np.concatenate((r_channel, g_channel, b_channel), axis=2)
    # cv2.imwrite(f'D:\\Code\\VideoHDR-mm\\example-delta\\{name}.png', img_rgb[:,:,::-1])
    # ori_img = img_rgb / 255.

    #----------------------------------------------------------------------------------------
    # cmp_list =[]
    # for i in range(3):
    #     img_hsv = colour.RGB_to_HSV(ori_img)
    #     print(img_hsv[0][0])
    #     img_h = img_hsv.copy()[..., 0]
    #     img_h += (0.0 + (i+1) * 0.1)
    #     # img_h = np.where(img_h > 1.0, img_h-1.0, img_h)
    #     img_h_rgb = colour.HSV_to_RGB(np.concatenate((img_h[..., None], img_hsv[...,1:]), axis=-1))
    #     print(img_h_rgb[0][0])
    #     cmp_list.append(img_h_rgb)
    #     new_name = name + '_h_' + str(0.0 + (i+1) * 0.1)
    #     cv2.imwrite(f'D:\\Code\\VideoHDR-mm\\example-delta\\{new_name}.png', img_h_rgb[:,:,::-1]*255.)

    # cmp_list =[]
    # for i in range(3):
    #     img_hsv = colour.RGB_to_HSV(ori_img)
    #     img_s = img_hsv.copy()[..., 1]
    #     img_s *= (1.0 + (i+1) * 0.1)
    #     # img_s = np.where(img_h > 1.0, img_h-1.0, img_h)
    #     img_s_rgb = colour.HSV_to_RGB(np.concatenate((img_hsv[..., 0][..., None], img_s[..., None], img_hsv[..., 2][..., None]), axis=-1))
    #     print(img_s_rgb[0][0])
    #     cmp_list.append(img_s_rgb)
    #     new_name = name + '_' + str(1.0 + (i+1) * 0.1)
    #     cv2.imwrite(f'D:\\Code\\VideoHDR-mm\\example-delta\\{new_name}.png', img_s_rgb[:,:,::-1]*255.)


    # cmp_list =[]
    # for i in range(3):
    #     img_hsv = colour.RGB_to_HSV(ori_img)
    #     print(img_hsv[0][0])
    #     img_v = img_hsv.copy()[..., 2]
    #     img_v *= (1.0 + (i+1) * 0.1)
    #     img_v = np.where(img_v > 1.0, img_v-1.0, img_v)
    #     img_v_rgb = colour.HSV_to_RGB(np.concatenate((img_hsv[...,:2], img_v[..., None]), axis=-1))
    #     print(img_v_rgb[0][0])
    #     cmp_list.append(img_v_rgb)
    #     new_name = name + '_v_' + str(1.0 + (i+1) * 0.1)
        # cv2.imwrite(f'D:\\Code\\VideoHDR-mm\\example-delta\\{new_name}.png', img_v_rgb[:,:,::-1]*255.)
    #-----------------------------------------------------------------------------------------

    # res = DeltaE_cmp.cmp(ori_img, cmp_list)
    # print(res)
    # cv2.imwrite(f'D:\\Code\\VideoHDR-mm\\example-delta\\red_h.png', img_rgb[:,:,::-1]*255.)


    a = np.array([0.2,0.2,0.2])
    b = np.array([20,20,20])
    c = np.array([0.1,0.1,0.1])
    d = np.array([10,10,10])
    print(DeltaE_cmp.delta_E_CIE2000(a, c))

    sdr = (cv2.imread(r"D:\Code\VideoHDR-mm\example-data\01316.png") / 255.)[:,:,::-1]
    hdr = cv2.imread(r"D:\Code\VideoHDR-mm\example-data\01316_hdr.png", cv2.IMREAD_UNCHANGED)[:,:,::-1] / 65535.
    hdrxy = cv2.imread(r"D:\Code\VideoHDR-mm\example-data\01316_hdr_delta_xy.png", cv2.IMREAD_UNCHANGED)[:,:,::-1] / 65535.
    sdr_0 = DeltaE_cmp.cmp(sdr, [hdr], ori_type='709',cmp_type='2020')
    sdr_1 = DeltaE_cmp.cmp(sdr, [hdrxy], ori_type='709',cmp_type='2020')

    print(sdr_0, sdr_1)

 



