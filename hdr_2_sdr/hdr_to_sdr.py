import math
import torch

from color import color_model
from gamut_compress.twp_xyY import Gamut_Conversion
# from gamut_compress.closest_xyY import Gamut_Conversion
# from gamut_compress.closest_uvY import Gamut_Conversion

class SdrConversion:
    def __init__(self, dark_point=(15, 18), skin_point=(63, 110), reference_point=(90.667, 203), sdr_ip=70,
                 high_light=(100, 400), a=120, b=0, alpha=1000, beta=0, cuda_available=True):
        """
        :param dark_point:
        :param skin_point:
        :param reference_point:
        :param sdr_ip:
        :param a:
        :param b:
        :param alpha:
        :param beta:
        """
        self.sdr_dark, self.hdr_dark = dark_point
        self.sdr_skin, self.hdr_skin = skin_point
        self.sdr_ref, self.hdr_ref = reference_point
        self.sdr_high, self.hdr_high = high_light
        self.sdr_ip = sdr_ip
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.compute_tone_mapping_parameters()
        self.cuda_available = cuda_available
        self.RGB709_to_XYZ_matrix =  torch.tensor([[0.412391, 0.357584, 0.180481],
                                                  [0.212639, 0.715169, 0.072192],
                                                  [0.019331, 0.119195, 0.950532]], dtype= torch.float32)
        
        self.XYZ_to_RGB709_matrix = torch.tensor([[ 3.2409663 , -1.5373788 , -0.49861172],
                                                    [-0.96924204,  1.8759652 ,  0.04155577],
                                                    [ 0.05562956, -0.20397693,  1.0569717 ]], dtype=torch.float32)

        self.RGB2020_to_XYZ_matrix =  torch.tensor([[0.636958, 0.144617, 0.168881],
                                                   [0.262700, 0.677998, 0.059302],
                                                   [0.000000, 0.028073, 1.060985]], dtype= torch.float32)

        self.XYZ_to_RGB2020_matrix =  torch.tensor([[1.716651, -0.355671, -0.253366],
                                                   [-0.666684, 1.616481, 0.015769],
                                                   [0.017640, -0.042771, 0.942103]], dtype= torch.float32)

        self.RGB709_to_RGB2020_matrix =  torch.tensor([[0.627404, 0.329282, 0.043314],
                                                      [0.069097, 0.919541, 0.011362],
                                                      [0.016392, 0.088013, 0.895595]], dtype= torch.float32)

        self.RGB2020_to_RGB709_matrix =  torch.tensor([[1.660491, -0.587641, -0.072850],
                                                      [-0.124551, 1.132900, -0.008349],
                                                      [-0.018151, -0.100579, 1.118730]], dtype= torch.float32)

        if self.cuda_available:
            self.RGB709_to_XYZ_matrix = self.RGB709_to_XYZ_matrix.cuda()
            self.RGB2020_to_XYZ_matrix = self.RGB2020_to_XYZ_matrix.cuda()
            self.XYZ_to_RGB2020_matrix = self.XYZ_to_RGB2020_matrix.cuda()
            self.RGB709_to_RGB2020_matrix = self.RGB709_to_RGB2020_matrix.cuda()
            self.RGB2020_to_RGB709_matrix = self.RGB2020_to_RGB709_matrix.cuda()

        self.Gamut_Conversion = Gamut_Conversion()

    def compute_tone_mapping_parameters(self):
        k0 = self.sdr_dark / self.hdr_dark
        k1 = (self.sdr_skin - self.sdr_dark) / (self.hdr_skin - self.hdr_dark)
        b1 = self.sdr_dark - k1 * self.hdr_dark
        self.hdr_ip = (self.sdr_ip - b1) / k1

        min_value, k3_res = 100000, 100000
        for i in range(20000):
            k3 = (i - 10000) / 10000
            k2 = (1 - k3) * self.sdr_ip
            k4 = self.sdr_ip - k2 * math.log(1 - k3)

            res = abs(self.sdr_ref - k2 * math.log(self.hdr_ref / self.hdr_ip - k3) - k4)
            if res < min_value:
                min_value, k3_res = res, k3
        k2 = (1 - k3_res) * self.sdr_ip
        k4 = self.sdr_ip - k2 * math.log(1 - k3_res)
        k3 = k3_res
        return k0, k1, b1, k2, k3, k4

    def tone_mapping(self, Y_hdr):
        """
        apply inverse tone mapping in sdr Y domain
        :param Y_sdr: sdr image linear Y data
        :return: Y_hdr: hdr image linear Y data
        """
        k0, k1, b1, k2, k3, k4 = self.compute_tone_mapping_parameters()
        Y_sdr =  torch.empty_like(Y_hdr)
        Y_sdr =  torch.where(Y_hdr <= self.hdr_dark, k0 * Y_hdr, Y_sdr)
        Y_sdr =  torch.where((Y_hdr > self.hdr_dark) & (Y_hdr < self.hdr_ip), k1 * Y_hdr + b1, Y_sdr)
        Y_sdr =  torch.where(Y_hdr >= self.hdr_ip, k2 *  torch.log(Y_hdr / self.hdr_ip - k3) + k4, Y_sdr)
        return Y_sdr

    def hdr_to_sdr(self, image_rgb):
        """
        convent sdr image to hdr image
        :param image_rgb: sdr rgb image with range (0~1)
        :return: image_rgb_hdr_nolinear: hdr rgb image with range (0~1)
        """
        if self.cuda_available:
            image_rgb = image_rgb.cuda()

        hdr_RGB2020_nolinear = image_rgb
        # step2  SDR EOTF  use BT.1886
        hdr_RGB2020_linear = color_model.eotf_HLG_BT2100(hdr_RGB2020_nolinear, self.alpha, self.beta)
        # step3  apply crosstalk matrix
        hdr_xRGB2020_linear = hdr_RGB2020_linear
        image_xyz =  torch.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, hdr_xRGB2020_linear)
        image_xyY = color_model.XYZ_to_xyY(image_xyz)
        Y_sdr = self.tone_mapping(image_xyY[:, :, 2])
        sdr_xyY = image_xyY.clone()
        sdr_xyY[:, :, 2] = Y_sdr
        sdr_XYZ = color_model.xyY_to_XYZ(sdr_xyY)

        image_rgb_sdr =  torch.einsum('ic,hwc->hwi', self.XYZ_to_RGB2020_matrix, sdr_XYZ)
        image_rgb_sdr[image_rgb_sdr < 0] = 0

        #-------------------- EOTF-1--------------------------------------
        if self.b == 0:
            sdr_RGB2020_nolinear = (image_rgb_sdr / self.a) ** (1 / 2.4)
        else:
            sdr_RGB2020_nolinear = 0

        sdr_RGB2020_nolinear =  torch.where(torch.isnan(sdr_RGB2020_nolinear),  torch.zeros_like(sdr_RGB2020_nolinear),
                                       sdr_RGB2020_nolinear)   
        
        #--------------------BT.2020 -> BT.709----------------------------------
        # result_rgb_image_v0 =  torch.einsum('ic,hwc->hwi', self.RGB2020_to_RGB709_matrix, sdr_RGB2020_nolinear)
        result_rgb_image_v1 = self.Gamut_Conversion.gamut_twp_xyY(sdr_RGB2020_nolinear)
        # result_rgb_image_v2 = self.Gamut_Conversion.gamut_closet_xyY(sdr_RGB2020_nolinear)
        # result_rgb_image_v3 = self.Gamut_Conversion.gamut_closet_uvY(sdr_RGB2020_nolinear)

        # #--TODO
        result_rgb_image_v1[result_rgb_image_v1 > 1] = 1
        result_rgb_image_v1[result_rgb_image_v1 < 0] = 0
        return result_rgb_image_v1

        # #-------------------test color difference----------------------------------------------------------------
        # res_list = [result_rgb_image_v0, result_rgb_image_v1, result_rgb_image_v2, result_rgb_image_v3]
        # for x in res_list:
        #     x[x < 0] = 0 
        #     x[x > 1] = 1
        # deltaE_list = self.DeltaE_cmp.cmp(hdr_RGB2020_nolinear, res_list, ori_type='2020', cmp_type='709')
        # deltaE_list_with_v0 = self.DeltaE_cmp.cmp(result_rgb_image_v0, res_list, ori_type='709', cmp_type='709')

        # return res_list, deltaE_list, deltaE_list_with_v0

