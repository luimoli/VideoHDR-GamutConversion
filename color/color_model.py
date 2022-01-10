import torch 

def stack(arr):
    return torch.cat([x[..., None] for x in arr], axis=-1)

def split(arr):
    return [arr[..., x] for x in range(arr.shape[-1])]

def xyY_to_XYZ(xyY):
    """
    [Converts between *CIE XYZ* tristimulus values and *CIE xyY* colourspace with reference *illuminant*.]
    Args:
        xyY ([array_like]): [*CIE xyY* colourspace array in domain [0, 1].]
    Returns:
        [array_like]: [*CIE XYZ* tristimulus values array in domain [0, 1].]
    """
    x, y, Y = split(xyY)
    XYZ = torch.where((y == 0)[..., None], stack((y, y, y)), stack((x * Y / y, Y, (1 - x - y) * Y / y)))

    return XYZ


def XYZ_to_xyY(XYZ, illuminant=[ 0.3127,  0.329]):
    """
    [Converts from *CIE XYZ* tristimulus values to *CIE xyY* colourspace with reference *illuminant*.]
    Args:
        XYZ ([array_like]): [*CIE XYZ* tristimulus values in domain [0, 1].]
    Returns:
        [type]: [*CIE xyY* colourspace array in domain [0, 1].]
    """
    X, Y, Z = split(XYZ)
    XYZ_n = torch.zeros(XYZ.shape)
    XYZ_n[..., 0:2] = torch.tensor(illuminant)

    # replace the point which contains 0 in XYZ-format to avoid zero-divide.
    xyY = torch.where(torch.all(XYZ == 0, axis=-1)[..., None], XYZ_n, stack((X / (X + Y + Z), Y / (X + Y + Z), Y))) 

    return xyY

def XYZ_to_uvY(XYZ):
    '''
    from XYZ to u'v'Y
    '''
    uvY =  torch.empty_like(XYZ)
    X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]
    uvY[:, :, 2] = Y
    uvY[:, :, 0] = 4*X / (X + 15*Y + 3*Z)
    uvY[:, :, 1] = 9*Y / (X + 15*Y + 3*Z)
    return uvY

def uvY_to_XYZ(uvY):
    '''
    from u'v'Y to XYZ
    '''
    XYZ =  torch.empty_like(uvY, dtype=torch.float32)
    u, v, Y = uvY[:, :, 0], uvY[:, :, 1], uvY[:, :, 2]
    XYZ[:, :, 1] = Y
    XYZ[:, :, 0] = Y * ((u * 9) / (v * 4))
    XYZ[:, :, 2] = Y* ((12 - 3*u - 20*v) / (4*v))
    return XYZ


def hlg_inverse_ootf(rgb_screen, alpha=1000, beta=0):
    Yd = 0.2627 * (rgb_screen[:, :, 0:1]) + 0.6780 * (rgb_screen[:, :, 1:2]) + 0.0593 * (rgb_screen[:, :, 2:])
    gamma = 1.2
    rgb_scene = ((Yd - beta) / alpha) ** ((1 - gamma) / gamma) * ((rgb_screen - beta) / alpha)
    return rgb_scene


def hlg_oetf(rgb2100):
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    rgb_oetf =  torch.where(rgb2100 <= 1 / 12, ((3 * rgb2100) ** 0.5), (a *  torch.log(12 * rgb2100 - b) + c))
    return rgb_oetf

def hlg_inverse_eotf(data, alpha, beta):
    step1 = hlg_inverse_ootf(data, alpha, beta)
    return hlg_oetf(step1)


def eotf_HLG_BT2100(rgb2020, L_W=1000, L_B=0):
    step1 = oetf_inverse_ARIBSTDB67(rgb2020) / 12.
    return ootf_HLG_BT2100_1(step1, L_B, L_W)


def ootf_HLG_BT2100_1(x, L_B, L_W, gamma=1.2):
    R_S, G_S, B_S = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    alpha = L_W - L_B
    beta = L_B
    Y_S = 0.2627 * R_S + 0.6780 * G_S + 0.0593 * B_S

    R_D = alpha * R_S *  torch.abs(Y_S) ** (gamma - 1) + beta
    G_D = alpha * G_S *  torch.abs(Y_S) ** (gamma - 1) + beta
    B_D = alpha * B_S *  torch.abs(Y_S) ** (gamma - 1) + beta

    RGB_D =  torch.stack([R_D, G_D, B_D], axis=2)
    return RGB_D


def oetf_inverse_ARIBSTDB67(E_p):
    # #TODO! ---should verify E_p 's range: 0.5 or 1?
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    E =  torch.where(E_p <= 1, (E_p / 0.5) ** 2,  torch.exp((E_p - c) / a) + b)
    return E
