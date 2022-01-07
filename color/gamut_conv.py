import colour
import numpy as np
from color import color_model
import matplotlib.pyplot as plt

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
        self.hdr_prima, self.sdr_prima, self.white_point, self.uv_white_point, self.uv_sdr_prima, self.uv_hdr_prima = self.compute_prima()

    def xy_to_uv(self, x, y):
        u = (4*x) / (3-2*x+12*y)
        v = (9*y) / (3-2*x+12*y)
        return np.concatenate((u[...,None], v[...,None]),-1)
    
    def compute_prima(self):
        hdr_prima = np.array([[ 0.708,  0.292],
                                [ 0.17 ,  0.797],
                                [ 0.131,  0.046]])
        sdr_prima = np.array([[ 0.64,  0.33],
                                [ 0.3 ,  0.6 ],
                                [ 0.15,  0.06]])
        white_point = np.array([ 0.3127,  0.329 ])
        uv_white_point = self.xy_to_uv(white_point[0], white_point[1])
        uv_sdr_prima =  self.xy_to_uv(sdr_prima[:,0], sdr_prima[:,1])
        uv_hdr_prima =  self.xy_to_uv(hdr_prima[:,0], hdr_prima[:,1])
        return hdr_prima, sdr_prima, white_point, uv_white_point, uv_sdr_prima, uv_hdr_prima
    
    def calc_k_b(self, x1, y1, x2, y2):
        k = (y2-y1) / (x2-x1)
        b = y1 - k*x1
        return (k, b)

    def calc_cross_point(self, k1,b1, k2,b2):
        x = (b2-b1) / (k1-k2)
        y = k2 * x + b2
        return x, y
        # return np.concatenate((x[..., None], y[..., None]), axis=-1)
    def euclidean(self, x1, y1, x2, y2): 
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def func_corner_closet(self, k1, b1, k2, b2, rgb_peak, sdrx, sdry, corner_type):
        '''
        choose the points of the intersection angle formed by two lines.
        k1,b1: line 1
        k2,b2: line 2
        param rgb_peak : the peak of the corner
        sdrx,sdry: the points
        cornertype: choose which section formed by two lines
        '''
        if corner_type == '<<':
            sdrx_cut = np.where((k1 * sdrx + b1 < sdry) & ( k2 * sdrx + b2 < sdry), 
                        np.ones_like(sdry)*rgb_peak[0], sdrx)
            sdry_cut = np.where((k1 * sdrx + b1 < sdry) & ( k2 * sdrx + b2 < sdry), 
                        np.ones_like(sdry)*rgb_peak[1], sdry)
        elif corner_type == '<>':
            sdrx_cut = np.where((k1 * sdrx + b1 < sdry) & ( k2 * sdrx + b2 > sdry), 
                        np.ones_like(sdry)*rgb_peak[0], sdrx)
            sdry_cut = np.where((k1 * sdrx + b1 < sdry) & ( k2 * sdrx + b2 > sdry), 
                        np.ones_like(sdry)*rgb_peak[1], sdry)
        elif corner_type == '><':
            sdrx_cut = np.where((k1 * sdrx + b1 > sdry) & ( k2 * sdrx + b2 < sdry), 
                        np.ones_like(sdry)*rgb_peak[0], sdrx)
            sdry_cut = np.where((k1 * sdrx + b1 > sdry) & ( k2 * sdrx + b2 < sdry), 
                        np.ones_like(sdry)*rgb_peak[1], sdry)
        elif corner_type == '>>':
            sdrx_cut = np.where((k1 * sdrx + b1 > sdry) & ( k2 * sdrx + b2 > sdry), 
                        np.ones_like(sdry)*rgb_peak[0], sdrx)
            sdry_cut = np.where((k1 * sdrx + b1 > sdry) & ( k2 * sdrx + b2 > sdry), 
                        np.ones_like(sdry)*rgb_peak[1], sdry)
        else:
            print('wrong corner type!')
        return sdrx_cut, sdry_cut

    def calc_vertical_cross_points(self, k, b, x, y):
        '''
        given point(x, y), get the vertical cross point of the line(determined by 'k' and 'b').
        '''
        k_v = (-1.0) / k
        b_v = y - k_v * x
        res_x, res_y = self.calc_cross_point(k_v, b_v, k, b)
        return res_x, res_y

    def gamut_twp_xyY(self, sdr_RGB2020_signal):
        sdr2020_xyz =  np.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, sdr_RGB2020_signal)
        sdr2020_xyY = color_model.XYZ_to_xyY(sdr2020_xyz)
        sdr2020_xyY =  np.where( np.isnan(sdr2020_xyY),  np.zeros_like(sdr2020_xyY), sdr2020_xyY)  #TODO
        sdrx, sdry = sdr2020_xyY[:, :, 0], sdr2020_xyY[:, :, 1]
        
        rg_line_k, rg_line_b = self.calc_k_b(*self.sdr_prima[0], *self.sdr_prima[1])
        rb_line_k, rb_line_b = self.calc_k_b(*self.sdr_prima[0], *self.sdr_prima[2])
        gb_line_k, gb_line_b = self.calc_k_b(*self.sdr_prima[1], *self.sdr_prima[2])
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

        # plt.figure(num=3, figsize=(8, 8))
        # plt.scatter(sdrx_cut2,sdry_cut2,s=1,c='r')
        # plt.scatter(sdr2020_xyY[:,:,0],sdr2020_xyY[:,:,1],s=0.1,c='b')
        # plt.show()
        # import ipdb;ipdb.set_trace()

        sdr709_xyY =  np.concatenate((sdrx_cut2[..., None], sdry_cut2[..., None], sdr2020_xyY[:,:,2].copy()[..., None]), axis=-1)
        sdr709_xyz = color_model.xyY_to_XYZ(sdr709_xyY)
        sdr_RGB709 = np.einsum('ic,hwc->hwi', self.XYZ_to_RGB709_matrix, sdr709_xyz)
        return sdr_RGB709

    def gamut_twp_uvY(self, sdr_RGB2020_signal):
        sdr2020_xyz =  np.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, sdr_RGB2020_signal)
        sdr2020_uvY = color_model.XYZ_to_uvY(sdr2020_xyz)
        sdr2020_uvY =  np.where( np.isnan(sdr2020_uvY),  np.zeros_like(sdr2020_uvY), sdr2020_uvY)  #TODO
        sdrx, sdry = sdr2020_uvY[:, :, 0], sdr2020_uvY[:, :, 1]
        
        rg_line_k, rg_line_b = self.calc_k_b(*self.uv_sdr_prima[0], *self.uv_sdr_prima[1])
        rb_line_k, rb_line_b = self.calc_k_b(*self.uv_sdr_prima[0], *self.uv_sdr_prima[2])
        gb_line_k, gb_line_b = self.calc_k_b(*self.uv_sdr_prima[1], *self.uv_sdr_prima[2])
        wk, wb = self.calc_k_b(sdrx, sdry, *self.uv_white_point)
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
        
        sdrx_cut2 = np.where(gb_line_k * sdrx_cut1 + gb_line_b > sdry_cut1, 
                            cross_gb[0], sdrx_cut1)
        sdry_cut2 = np.where(gb_line_k * sdrx_cut1 + gb_line_b > sdry_cut1, 
                            cross_gb[1], sdry_cut1)

        # plt.figure(num=3, figsize=(8, 8))
        # plt.scatter(sdrx_cut2,sdry_cut2,s=1,c='coral')
        # # plt.scatter(sdr2020_uvY[:,:,0],sdr2020_uvY[:,:,1],s=0.1,c='cornflowerblue')
        # plt.scatter(self.uv_white_point[0], self.uv_white_point[1], s=4, c = 'black', label='white point')
        # plt.plot(self.uv_sdr_prima[:,0], self.uv_sdr_prima[:,1], color='r', label= 'bt.709')
        # plt.plot(self.uv_hdr_prima[:,0], self.uv_hdr_prima[:,1], color='b', label= 'bt.2020')
        # plt.plot((self.uv_sdr_prima[:,0][0], self.uv_sdr_prima[:,0][2]), (self.uv_sdr_prima[:,1][0],self.uv_sdr_prima[:,1][2]), color='r')
        # plt.plot((self.uv_hdr_prima[:,0][0], self.uv_hdr_prima[:,0][2]), (self.uv_hdr_prima[:,1][0],self.uv_hdr_prima[:,1][2]), color='b')
        # plt.legend()
        # import ipdb;ipdb.set_trace()
        # plt.show()

        sdr709_uvY =  np.concatenate((sdrx_cut2[..., None], sdry_cut2[..., None], sdr2020_uvY[:,:,2].copy()[..., None]), axis=-1)
        sdr709_xyz = color_model.uvY_to_XYZ(sdr709_uvY)
        sdr_RGB709 = np.einsum('ic,hwc->hwi', self.XYZ_to_RGB709_matrix, sdr709_xyz)

        return sdr_RGB709

    def gamut_closet_xyY(self, sdr_RGB2020_signal):
        sdr2020_xyz =  np.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, sdr_RGB2020_signal)
        sdr2020_xyY = color_model.XYZ_to_xyY(sdr2020_xyz)
        sdr2020_xyY =  np.where( np.isnan(sdr2020_xyY),  np.zeros_like(sdr2020_xyY), sdr2020_xyY)  #TODO
        sdrx, sdry = sdr2020_xyY[:, :, 0],sdr2020_xyY[:, :, 1]
        Rc, Gc, Bc = self.sdr_prima
        rg_709_k, rg_709_b = self.calc_k_b(*self.sdr_prima[0],*self.sdr_prima[1])
        rb_709_k, rb_709_b = self.calc_k_b(*self.sdr_prima[0],*self.sdr_prima[2])
        gb_709_k, gb_709_b = self.calc_k_b(*self.sdr_prima[1],*self.sdr_prima[2])

        # ------ vertical lines ------------
        rg_k, rg_b = (-1.0 / rg_709_k), Rc[1] - Rc[0] * (-1.0 / rg_709_k)
        gr_k, gr_b = rg_k, Gc[1] - Gc[0] * rg_k

        rb_k, rb_b = (-1.0 / rb_709_k), Rc[1] - Rc[0] * (-1.0 / rb_709_k)
        br_k, br_b = rb_k, Bc[1] - Bc[0] * rb_k

        gb_k, gb_b = (-1.0 / gb_709_k), Gc[1] - Gc[0] * (-1.0 / gb_709_k)
        bg_k, bg_b = gb_k, Bc[1] - Bc[0] * gb_k

        sdrx_cut, sdry_cut = sdrx.copy(), sdry.copy()
        sdrx_cut, sdry_cut = self.func_corner_closet(gb_k, gb_b, gr_k, gr_b, Gc, sdrx_cut, sdry_cut, corner_type='<<')
        sdrx_cut, sdry_cut = self.func_corner_closet(bg_k, bg_b, br_k, br_b, Bc, sdrx_cut, sdry_cut, corner_type='>>')
        sdrx_cut, sdry_cut = self.func_corner_closet(rb_k, rb_b, rg_k, rg_b, Rc, sdrx_cut, sdry_cut, corner_type='<>')

        p709_rg =  self.calc_vertical_cross_points(rg_709_k, rg_709_b, sdrx_cut, sdry_cut)
        p709_rb =  self.calc_vertical_cross_points(rb_709_k, rb_709_b, sdrx_cut, sdry_cut)
        p709_gb =  self.calc_vertical_cross_points(gb_709_k, gb_709_b, sdrx_cut, sdry_cut)

        sdrx_cut = np.where(rg_709_k * sdrx_cut + rg_709_b  < sdry_cut, p709_rg[0], sdrx_cut)
        sdry_cut = np.where(rg_709_k * sdrx_cut + rg_709_b  < sdry_cut, p709_rg[1], sdry_cut)
        sdrx_cut = np.where(gb_709_k * sdrx_cut + gb_709_b  < sdry_cut, p709_gb[0], sdrx_cut)
        sdry_cut = np.where(gb_709_k * sdrx_cut + gb_709_b  < sdry_cut, p709_gb[1], sdry_cut)
        sdrx_cut = np.where(rb_709_k * sdrx_cut + rb_709_b  > sdry_cut, p709_rb[0], sdrx_cut)
        sdry_cut = np.where(rb_709_k * sdrx_cut + rb_709_b  > sdry_cut, p709_rb[1], sdry_cut)

        # plt.figure(num=3, figsize=(8, 8))
        # plt.title("the distribution of x and y")
        # plt.xlabel('CIE x')
        # plt.ylabel('CIE y')
        # plt.plot(sdr_prima[:,0], sdr_prima[:,1], color='r')
        # plt.plot((sdr_prima[:,0][0], sdr_prima[:,0][2]), (sdr_prima[:,1][0],sdr_prima[:,1][2]), color='r')
        # plt.plot(hdr_prima[:,0], hdr_prima[:,1], color='b')
        # plt.plot((hdr_prima[:,0][0], hdr_prima[:,0][2]), (hdr_prima[:,1][0],hdr_prima[:,1][2]), color='b')
        # plt.scatter(sdrx_cut,sdry_cut,s=0.1,c='r',label='bt.709')
        # # plt.scatter(sdr2020_xyY[:,:,0],sdr2020_xyY[:,:,1],s=0.1,c='b',label='bt.2020')
        
        sdr709_xyY =  np.concatenate((sdrx_cut[..., None], sdry_cut[..., None], sdr2020_xyY[:,:,2].copy()[..., None]), axis=-1)
        sdr709_xyz = color_model.xyY_to_XYZ(sdr709_xyY)
        sdr_RGB709 = np.einsum('ic,hwc->hwi', self.XYZ_to_RGB709_matrix, sdr709_xyz)

        # #--------------test saturation boundary---------------------------------------------------
        # sdr_hsv = colour.RGB_to_HSV(sdr_RGB709)
        # sdr_hsv[:,:,1] *= 1.1
        # sdr_RGB709_mores = colour.HSV_to_RGB(sdr_hsv)
        # sdr_RGB709_mores_xyz =  np.einsum('ic,hwc->hwi', RGB709_to_XYZ_matrix, sdr_RGB709_mores)
        # sdr_RGB709_mores_xyY = color_model.XYZ_to_xyY(sdr_RGB709_mores_xyz)
        # plt.scatter(sdr_RGB709_mores_xyY[:,:,0],sdr_RGB709_mores_xyY[:,:,1],s=0.1,c='g',label='bt.our_test_res')
        # plt.legend()
        # import ipdb;ipdb.set_trace()

        return sdr_RGB709
        # return sdr_RGB709_mores

    def gamut_closet_uvY(self, sdr_RGB2020_signal):
        sdr2020_xyz =  np.einsum('ic,hwc->hwi', self.RGB2020_to_XYZ_matrix, sdr_RGB2020_signal)
        sdr2020_uvY = color_model.XYZ_to_uvY(sdr2020_xyz)
        # sdr2020_uvY =  np.where( np.isnan(sdr2020_uvY),  np.zeros_like(sdr2020_uvY), sdr2020_uvY)  #TODO
        sdrx, sdry = sdr2020_uvY[:, :, 0],sdr2020_uvY[:, :, 1]
        Rc, Gc, Bc = self.uv_sdr_prima
        rg_709_k, rg_709_b = self.calc_k_b(*self.uv_sdr_prima[0],*self.uv_sdr_prima[1])
        rb_709_k, rb_709_b = self.calc_k_b(*self.uv_sdr_prima[0],*self.uv_sdr_prima[2])
        gb_709_k, gb_709_b = self.calc_k_b(*self.uv_sdr_prima[1],*self.uv_sdr_prima[2])

        # ------ vertical lines' k and b------------
        rg_k, rg_b = (-1.0 / rg_709_k), Rc[1] - Rc[0] * (-1.0 / rg_709_k)
        gr_k, gr_b = rg_k, Gc[1] - Gc[0] * rg_k

        rb_k, rb_b = (-1.0 / rb_709_k), Rc[1] - Rc[0] * (-1.0 / rb_709_k)
        br_k, br_b = rb_k, Bc[1] - Bc[0] * rb_k

        gb_k, gb_b = (-1.0 / gb_709_k), Gc[1] - Gc[0] * (-1.0 / gb_709_k)
        bg_k, bg_b = gb_k, Bc[1] - Bc[0] * gb_k

        # ------- process corner points--------------
        sdrx_cut, sdry_cut = sdrx.copy(), sdry.copy()
        sdrx_cut, sdry_cut = self.func_corner_closet(gb_k, gb_b, gr_k, gr_b, Gc, sdrx_cut, sdry_cut, corner_type='<<')
        sdrx_cut, sdry_cut = self.func_corner_closet(bg_k, bg_b, br_k, br_b, Bc, sdrx_cut, sdry_cut, corner_type='>>')
        sdrx_cut, sdry_cut = self.func_corner_closet(rb_k, rb_b, rg_k, rg_b, Rc, sdrx_cut, sdry_cut, corner_type='<>')

        # ------- get vertical cross points (closest)
        p709_rg =  self.calc_vertical_cross_points(rg_709_k, rg_709_b, sdrx_cut, sdry_cut)
        p709_rb =  self.calc_vertical_cross_points(rb_709_k, rb_709_b, sdrx_cut, sdry_cut)
        p709_gb =  self.calc_vertical_cross_points(gb_709_k, gb_709_b, sdrx_cut, sdry_cut)
        sdrx_cut = np.where(rg_709_k * sdrx_cut + rg_709_b  < sdry_cut, p709_rg[0], sdrx_cut)
        sdry_cut = np.where(rg_709_k * sdrx_cut + rg_709_b  < sdry_cut, p709_rg[1], sdry_cut)
        sdrx_cut = np.where(gb_709_k * sdrx_cut + gb_709_b  > sdry_cut, p709_gb[0], sdrx_cut)
        sdry_cut = np.where(gb_709_k * sdrx_cut + gb_709_b  > sdry_cut, p709_gb[1], sdry_cut)
        sdrx_cut = np.where(rb_709_k * sdrx_cut + rb_709_b  > sdry_cut, p709_rb[0], sdrx_cut)
        sdry_cut = np.where(rb_709_k * sdrx_cut + rb_709_b  > sdry_cut, p709_rb[1], sdry_cut)

        # plt.figure(num=3, figsize=(8, 8))
        # plt.title("the distribution of u' and y'")
        # plt.xlabel('CIE u\'')
        # plt.ylabel('CIE v\'')
        # # plt.scatter(self.uv_white_point[0], self.uv_white_point[1], s=4, c = 'black', label='white point')
        # plt.plot(self.uv_sdr_prima[:,0], self.uv_sdr_prima[:,1], color='r', label= 'bt.709')
        # plt.plot(self.uv_hdr_prima[:,0], self.uv_hdr_prima[:,1], color='b', label= 'bt.2020')
        # plt.plot((self.uv_sdr_prima[:,0][0], self.uv_sdr_prima[:,0][2]), (self.uv_sdr_prima[:,1][0],self.uv_sdr_prima[:,1][2]), color='r')
        # plt.plot((self.uv_hdr_prima[:,0][0], self.uv_hdr_prima[:,0][2]), (self.uv_hdr_prima[:,1][0],self.uv_hdr_prima[:,1][2]), color='b')
        # # plt.scatter(sdrx_cut,sdry_cut,s=0.1,c='coral')
        # plt.scatter(sdr2020_uvY[:,:,0],sdr2020_uvY[:,:,1],s=0.1,c='cornflowerblue')
        # plt.legend()
        # import ipdb;ipdb.set_trace()
        # plt.show()
        
        sdr709_uvY =  np.concatenate((sdrx_cut[..., None], sdry_cut[..., None], sdr2020_uvY[:,:,2].copy()[..., None]), axis=-1)
        sdr709_xyz = color_model.uvY_to_XYZ(sdr709_uvY)
        sdr_RGB709 = np.einsum('ic,hwc->hwi', self.XYZ_to_RGB709_matrix, sdr709_xyz)
        return sdr_RGB709


#------------------------------------------------------------------------------------------------------------------

def func_lpcc(rn, beta, alpha):
    res = np.empty_like(rn)
    res = np.where(rn <= 1-beta, rn, res)
    res = np.where((rn > 1-beta) & (rn <= 1+alpha), 
                    rn - (alpha / (beta - alpha)**2)*(beta - np.sqrt(beta**2 + (alpha - beta)*(rn + beta -1))), res)
    res = np.where(rn > 1+alpha, 1, res)
    return res

def gamut_soft_clipping(sdr_RGB2020_signal):
    sdr2020_xyz =  np.einsum('ic,hwc->hwi', RGB2020_to_XYZ_matrix, sdr_RGB2020_signal)
    sdr2020_xyY = color_model.XYZ_to_xyY(sdr2020_xyz)
    # sdr2020_xyY =  np.where( np.isnan(sdr2020_xyY),  np.zeros_like(sdr2020_xyY),
    #                                    sdr2020_xyY) 
    sdrx = sdr2020_xyY[:, :, 0]
    sdry = sdr2020_xyY[:, :, 1]
    
    rg_709_k, rg_709_b = self.calc_k_b(sdr_prima[0][0], sdr_prima[0][1],sdr_prima[1][0], sdr_prima[1][1])
    rb_709_k, rb_709_b = self.calc_k_b(sdr_prima[0][0], sdr_prima[0][1],sdr_prima[2][0], sdr_prima[2][1])
    gb_709_k, gb_709_b = self.calc_k_b(sdr_prima[1][0], sdr_prima[1][1],sdr_prima[2][0], sdr_prima[2][1])

    gb_709_k2, gb_709_b2 = ( -1.0 / gb_709_k), white_point[1] - white_point[0] *(-1.0 / gb_709_k)

    rg_2020_k, rg_2020_b = self.calc_k_b(*hdr_prima[0],*hdr_prima[1])
    rb_2020_k, rb_2020_b = self.calc_k_b(*hdr_prima[0],*hdr_prima[2])
    gb_2020_k, gb_2020_b = self.calc_k_b(*hdr_prima[1],*hdr_prima[2])

    wk, wb = self.calc_k_b(*white_point, sdrx, sdry)
    p709 = self.calc_cross_point(wk, wb, gb_709_k, gb_709_b)
    p2020 = self.calc_cross_point(wk, wb, gb_2020_k, gb_2020_b)
    # w_p2020 = euclidean(*white_point, *p2020)
    # w_p709 = euclidean(*white_point, *p709)
    # w_cxy = euclidean(*white_point, sdrx, sdry)
    # alpha = w_p2020 / w_p709 - 1
    alpha = (p2020[0] - white_point[0]) / (p709[0] - white_point[0]) - 1
    beta = 0
    func_res = func_lpcc((sdrx - white_point[0]) / (p709[0] - white_point[0]), beta, alpha )
    func_res[ func_res < 0] = 0.0
    func_res[func_res > 1] = 1.0
    sdrx_cut = np.where(gb_709_k * sdrx + gb_709_b  < sdry, 
                (p709[0]-white_point[0]) * func_lpcc((sdrx - white_point[0]) / (p709[0] - white_point[0]), beta, alpha ) + white_point[0],
                # p709[0],
                sdrx)
    sdry_cut = np.where(gb_709_k * sdrx + gb_709_b  < sdry, 
                (p709[1]-white_point[1]) * func_lpcc((sdrx - white_point[0]) / (p709[0] - white_point[0]), beta, alpha ) + white_point[1],
                # p709[1],
                sdry)

    # test_func = np.where(gb_709_k * sdrx + gb_709_b  > sdry, 
    #             func_lpcc((sdrx - white_point[0]) / (p709[0] - white_point[0]), beta, alpha ),
    #             # p709[0],
    #             np.zeros_like(sdrx))

    # x = np.arange(0,0.4,0.1)
    plt.figure(num=3, figsize=(8, 8))
    plt.scatter(sdrx_cut,sdry_cut,s=0.5,c='r')
    # plt.scatter(sdr2020_xyY[:,:,0],sdr2020_xyY[:,:,1],s=0.1,c='b')
    plt.show()
    import ipdb;ipdb.set_trace()


    sdr709_xyY =  np.concatenate((sdrx_cut2[..., None], sdry_cut2[..., None], sdr2020_xyY[:,:,2].copy()[..., None]), axis=-1)
    sdr709_xyz = color_model.xyY_to_XYZ(sdr709_xyY)
    sdr_RGB709 = np.einsum('ic,hwc->hwi', XYZ_to_RGB709_matrix, sdr709_xyz)

    return sdr_RGB709

#--------------------------------------------------------------------------------------------



if __name__ == "__main__":
    gc = Gamut_Conversion()
    print(gc.white_point)
    print(gc.uv_white_point)
