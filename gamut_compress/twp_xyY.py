import numpy as np
from color import color_model

class Gamut_Conversion:
    def __init__(self) -> None:
        self.RGB709_to_XYZ_matrix =  np.array([[0.412391, 0.357584, 0.180481],
                                                    [0.212639, 0.715169, 0.072192],
                                                    [0.019331, 0.119195, 0.950532]], dtype= np.float32)
            
        self.XYZ_to_RGB709_matrix = np.array([[ 3.2409663 , -1.5373788 , -0.49861172],
                                                    [-0.96924204,  1.8759652 ,  0.04155577],
                                                    [ 0.05562956, -0.20397693,  1.0569717 ]], dtype=np.float32)

        self.RGB2020_to_XYZ_matrix =  np.array([[0.636958, 0.144617, 0.168881],
                                                    [0.262700, 0.677998, 0.059302],
                                                    [0.000000, 0.028073, 1.060985]], dtype= np.float32)

        self.XYZ_to_RGB2020_matrix =  np.array([[1.716651, -0.355671, -0.253366],
                                                    [-0.666684, 1.616481, 0.015769],
                                                    [0.017640, -0.042771, 0.942103]], dtype= np.float32)

        self.RGB709_to_RGB2020_matrix = np.array([[0.627404, 0.329282, 0.043314],
                                                      [0.069097, 0.919541, 0.011362],
                                                      [0.016392, 0.088013, 0.895595]], dtype=np.float32)

        self.BT2020_prima = np.array([[ 0.708,  0.292],
                                [ 0.17 ,  0.797],
                                [ 0.131,  0.046]], dtype=np.float32)

        self.BT709_prima = np.array([[ 0.64,  0.33],
                                [ 0.3 ,  0.6 ],
                                [ 0.15,  0.06]], dtype=np.float32)

        self.white_point = np.array([ 0.3127,  0.329 ], dtype=np.float32)

    def calc_k_b(self, x1, y1, x2, y2):
        '''
        calculate the k and b given two points (x1, y1) and (x2, y2).
        '''
        k = (y2-y1) / (x2-x1)
        b = y1 - k*x1
        return (k, b)

    def calc_cross_point(self, k1,b1, k2,b2):
        '''
        calculate the intersection point given two lines defined by (k1,b1) and (k2,b2).
        '''
        x = (b2-b1) / (k1-k2)
        y = k2 * x + b2
        return x, y

    def CG_BT2020_to_BT709(self, sdr_RGB2020_signal):
        '''
        this function turns SDR(bt.2020) into SDR(bt.709).
        ( used after HDR(bt.2020)_to_SDR(bt.2020) )

        :param: sdr_RGB2020_signal: RGB img of SDR(bt.2020), [0,1]
        :return: sdr_RGB709: RGB img of SDR(bt.709), [0,1]
        '''
        sdr2020_xyz =  np.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, sdr_RGB2020_signal)
        sdr2020_xyY = color_model.XYZ_to_xyY(sdr2020_xyz)
        sdr2020_xyY =  np.where( np.isnan(sdr2020_xyY),  np.zeros_like(sdr2020_xyY), sdr2020_xyY)  #TODO
        sdrx, sdry = sdr2020_xyY[:, :, 0], sdr2020_xyY[:, :, 1]
        
        rg_line_k, rg_line_b = self.calc_k_b(*self.BT709_prima[0], *self.BT709_prima[1])
        rb_line_k, rb_line_b = self.calc_k_b(*self.BT709_prima[0], *self.BT709_prima[2])
        gb_line_k, gb_line_b = self.calc_k_b(*self.BT709_prima[1], *self.BT709_prima[2])

        wk, wb = self.calc_k_b(sdrx, sdry, *self.white_point)
        cross_rg = self.calc_cross_point(wk, wb, rg_line_k,rg_line_b)
        cross_rb = self.calc_cross_point(wk, wb, rb_line_k, rb_line_b)
        cross_gb = self.calc_cross_point(wk, wb, gb_line_k, gb_line_b)

        sdrx_cut = np.where(rg_line_k * sdrx + rg_line_b < sdry, 
                            cross_rg[0], sdrx)
        sdry_cut = np.where(rg_line_k * sdrx + rg_line_b < sdry, 
                            cross_rg[1], sdry)

        sdrx_cut1 = np.where(rb_line_k * sdrx_cut + rb_line_b > sdry_cut, 
                            cross_rb[0], sdrx_cut)
        sdry_cut1 = np.where(rb_line_k * sdrx_cut + rb_line_b > sdry_cut, 
                            cross_rb[1], sdry_cut)
        
        sdrx_cut2 = np.where(gb_line_k * sdrx_cut1 + gb_line_b < sdry_cut1, 
                            cross_gb[0], sdrx_cut1)
        sdry_cut2 = np.where(gb_line_k * sdrx_cut1 + gb_line_b < sdry_cut1, 
                            cross_gb[1], sdry_cut1)

        sdr709_xyY =  np.concatenate((sdrx_cut2[..., None], sdry_cut2[..., None], sdr2020_xyY[:,:,2].copy()[..., None]), axis=-1)
        sdr709_xyz = color_model.xyY_to_XYZ(sdr709_xyY)
        sdr_RGB709 = np.einsum('ic,hwc->hwi', self.XYZ_to_RGB709_matrix, sdr709_xyz)

        sdr_RGB709[sdr_RGB709 > 1] = 1
        sdr_RGB709[sdr_RGB709 < 0] = 0

        return sdr_RGB709
    
    # def CG_BT709_to_BT2020(self, sdr_img_RGB709_nonlinear):
    #     '''
    #     this function turns SDR(bt.709) into SDR(bt.2020).
    #     ( used before SDR(bt.2020)_to_HDR(bt.2020) )

    #     :param: sdr_img_RGB709_nonlinear: RGB img of SDR(bt.2020), [0,1]
    #     :return: img_rgb2020_nolinear: RGB img of SDR(bt.709), [0,1]
    #     '''
    #     img_rgb709_linear = sdr_img_RGB709_nonlinear ** 2.4
    #     img_rgb2020_linear = np.einsum('ic,hwc->hwi', self.RGB709_to_RGB2020_matrix, img_rgb709_linear)
    #     img_rgb2020_nolinear = img_rgb2020_linear ** (1 / 2.4)
    #     return img_rgb2020_nolinear

if __name__ == "__main__":
    CG_conversion = Gamut_Conversion()
    print(CG_conversion.white_point)

    rgb709_example = np.float32(np.ones((540,960,3)) * 0.5)
    rgb2020_res = CG_conversion.CG_BT709_to_BT2020(rgb709_example)

    rgb2020_example = np.float32(np.ones((540,960,3)) * 0.5)
    rgb709_res = CG_conversion.CG_BT2020_to_BT709(rgb2020_example)
