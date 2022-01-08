import numpy as np 

def hlg_inverse_ootf(rgb_screen, alpha=1000, beta=0):
    Yd = 0.2627 * (rgb_screen[:, :, 0:1]) + 0.6780 * (rgb_screen[:, :, 1:2]) + 0.0593 * (rgb_screen[:, :, 2:])
    gamma = 1.2
    rgb_scene = ((Yd - beta) / alpha) ** ((1 - gamma) / gamma) * ((rgb_screen - beta) / alpha)
    return rgb_scene


def hlg_oetf(rgb2100):
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    rgb_oetf =  np.where(rgb2100 <= 1 / 12, ((3 * rgb2100) ** 0.5), (a *  np.log(12 * rgb2100 - b) + c))
    return rgb_oetf


def hlg_inverse_eotf(data, alpha, beta):
    step1 = hlg_inverse_ootf(data, alpha, beta)
    return hlg_oetf(step1)


def XYZ_to_xyY(XYZ):
    xyY =  np.empty_like(XYZ)
    X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]
    xyY[:, :, 2] = Y
    xyY[:, :, 0] = X / (X + Y + Z)
    xyY[:, :, 1] = Y / (X + Y + Z)
    return xyY

def xyY_to_XYZ(xyY):
    XYZ =  np.empty_like(xyY, dtype= np.float32)
    x, y, Y = xyY[:, :, 0], xyY[:, :, 1], xyY[:, :, 2]
    XYZ[:, :, 1] = Y
    XYZ[:, :, 0] = x * Y / y
    XYZ[:, :, 2] = (1 - x - y) * Y / y
    return XYZ


def XYZ_to_uvY(XYZ):
    '''
    from XYZ to u'v'Y
    '''
    uvY =  np.empty_like(XYZ)
    X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]
    uvY[:, :, 2] = Y
    uvY[:, :, 0] = 4*X / (X + 15*Y + 3*Z)
    uvY[:, :, 1] = 9*Y / (X + 15*Y + 3*Z)
    return uvY

def uvY_to_XYZ(uvY):
    '''
    from u'v'Y to XYZ
    '''
    XYZ =  np.empty_like(uvY, dtype=np.float32)
    u, v, Y = uvY[:, :, 0], uvY[:, :, 1], uvY[:, :, 2]
    XYZ[:, :, 1] = Y
    XYZ[:, :, 0] = Y * ((u * 9) / (v * 4))
    XYZ[:, :, 2] = Y* ((12 - 3*u - 20*v) / (4*v))
    return XYZ


def eotf_HLG_BT2100(rgb2020, L_W=1000, L_B=0):
    step1 = oetf_inverse_ARIBSTDB67(rgb2020) / 12.
    return ootf_HLG_BT2100_1(step1, L_B, L_W)


def ootf_HLG_BT2100_1(x, L_B, L_W, gamma=1.2):
    R_S, G_S, B_S = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    alpha = L_W - L_B
    beta = L_B
    Y_S = 0.2627 * R_S + 0.6780 * G_S + 0.0593 * B_S

    R_D = alpha * R_S *  np.abs(Y_S) ** (gamma - 1) + beta
    G_D = alpha * G_S *  np.abs(Y_S) ** (gamma - 1) + beta
    B_D = alpha * B_S *  np.abs(Y_S) ** (gamma - 1) + beta

    # RGB_D =  np.stack([R_D, G_D, B_D], dim=2)
    RGB_D =  np.stack([R_D, G_D, B_D], axis=2)
    return RGB_D


def oetf_inverse_ARIBSTDB67(E_p):
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    E =  np.where(E_p <= 1, (E_p / 0.5) ** 2,  np.exp((E_p - c) / a) + b)
    return E
