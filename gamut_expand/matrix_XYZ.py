import torch
from color import color_model

class CG_conversion:
    def __init__(self) -> None:
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

        self.RGB709_to_RGB2020_matrix = torch.tensor([[0.627404, 0.329282, 0.043314],
                                                      [0.069097, 0.919541, 0.011362],
                                                      [0.016392, 0.088013, 0.895595]], dtype=torch.float32)

        self.BT2020_prima = torch.tensor([[ 0.708,  0.292],
                                [ 0.17 ,  0.797],
                                [ 0.131,  0.046]], dtype=torch.float32)

        self.BT709_prima = torch.tensor([[ 0.64,  0.33],
                                [ 0.3 ,  0.6 ],
                                [ 0.15,  0.06]], dtype=torch.float32)

        self.white_point = torch.tensor([ 0.3127,  0.329 ], dtype=torch.float32)
    
    def CG_BT709_to_BT2020(self, sdr_img_RGB709_nonlinear):
        '''
        this function turns SDR(bt.709) into SDR(bt.2020).
        ( used before SDR(bt.2020)_to_HDR(bt.2020) )

        :param: sdr_img_RGB709_nonlinear: RGB img of SDR(bt.2020), [0,1]
        :return: img_rgb2020_nolinear: RGB img of SDR(bt.709), [0,1]
        '''
        img_rgb709_linear = sdr_img_RGB709_nonlinear ** 2.4
        img_rgb2020_linear = torch.einsum('ic,hwc->hwi', self.RGB709_to_RGB2020_matrix, img_rgb709_linear)
        img_rgb2020_nolinear = img_rgb2020_linear ** (1 / 2.4)
        return img_rgb2020_nolinear

if __name__ == "__main__":
    CG_conversion = CG_conversion()
    print(CG_conversion.white_point)

    rgb709_example = torch.ones((540,960,3), dtype=torch.float32) * 0.5
    rgb2020_res = CG_conversion.CG_BT709_to_BT2020(rgb709_example)
