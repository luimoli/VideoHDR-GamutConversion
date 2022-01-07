# VideoHDR
SDR/HDR conversion for video.   
This repository mainly contains two algorithms about SDR/HDR video format transfer: 1, converse SDR images to HDR images;
2, converse HDR videos to SDR videos.
## 1. **Introduction**
### 1.1 SDR and HDR   
Standard Dynamic Range content, which is publicly known as SDR, can be defined as a signal that is normally produced to be viewed
at a peak luminance of 100 cd/m2(nits) in the reference viewing environment, a black level close to zero and a bit-depth of 8 bits.   

As defined in Recommendation ITU-R BT.2100, High Dynamic Range Television(HDR-TV) provides viewers with an enhanced visual experience
by providing images that have been produced to look correct on brighter displays, that provide much brighter highlights, and that improved
details in dark areas.

The desire in such cases would be to increase the dynamic range of the content to effectively enhance its visual appearance. This requirement
may be translated into several objectives that any SDR to HDR conversion process should adhere to:  
(1) maintain details in the shadows;  
(2) ensure that mid-tones are not unduly expanded;  
(3) expand highlights up to the peak display luminance, insofar the quality of the content allows;  
(4) ensure chromatic content is adjusted appropriately;  
(5) maintain temporal stability.  

### 1.2 SDR_to_HDR  
This branch tells how to converse SDR images to HDR images. It mainly contains six steps, including:   
1, SDR BT.709 ---> SDR BT.2020  
2, EOTF(BT-1886)  
3, RGB --> XYZ --> xyY  
4, inverse tone mapping  
5, xyY --> XYZ --> RGB  
6, inverse_EOTF    

![sdr_to_hdr.png](https://i.loli.net/2021/07/27/4SJnwxj7qmycu3O.png)
### 1.3 HDR_to_SDR  
This branch tells how to converse SDR images to HDR images. It mainly contains six steps, including:   
1, EOTF_HLG  
2, RGB --> XYZ --> xyY  
3, tone mapping  
4, xyY --> XYZ --> RGB  
5, SDR_EOTF_inverse  
6, SDR BT.2020 ---> SDR BT.709  

![hdr_to_sdr.png](https://i.loli.net/2021/07/27/Z3Rc5pkuWVNqd7Y.png)

## 2. **Usage**   
### 2.1 prepare
Because this repository may use different color space transfer, please first git clone [SM-ColourBase](https://github.com/smartmore/SM-ColourBase).
Please organize the code folder like this:
```
   |--VideoHDR
      |--corlor
         |--sm_color.py
         |--     :
         |--     :
      |--decode_encode
         |--yuv_reader.py
         |--      :
         |--      :
      |--example-data
         |--hdr.png
         |--   :
         |--   :
      |--hdr
         |--sdr_to_hdr.py
         |--      :
         |--      :
```

### 2.2 sdr_to_hdr  
```
import cv2
import numpy as np
 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from sdr_to_hdr import HdrConversion

if __name__ == '__main__':
    HDRConvention = HdrConversion()
    image_path = r"../example-data/sdr.png"
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 255.
    image_rgb = np.float32(image_bgr[..., ::-1].copy())
    image_rgb =  np.from_numpy(image_rgb)
    image_rgb_hdr_nolinear = HDRConvention.sdr_to_hdr(image_rgb)
    image_rgb_hdr_nolinear = image_rgb_hdr_nolinear.cpu().detach().numpy()
    image_bgr_hdr_nolinear = image_rgb_hdr_nolinear[..., ::-1]
    cv2.imwrite(r"../example-data/hdr.png", np.uint16(image_bgr_hdr_nolinear * 65535.))
```  

#### Parameter initial:  
|                               |     type    | default value |           description           |
| :--------------------------:  |  :------:   |    :---:      |   :-------------------------:   | 
| parameter1(dark_point)        |    tuple    |    (15,18)    |hdr_dark scope from [15, 30]     | 
| parameter2(skin_point)        |    tuple    |   (63, 110)   |hdr_skin scope from [40, 140]    | 
| parameter3(reference_point)   |    tuple    | (90.667, 203) |--                               | 
| parameter4(sdr_ip)            |  float num  |       70      |--                               | 
| parameter5(high_light)        |    tuple    |   (100, 260)  |hdr_high scope from [260, 600]   | 
| a                             |  float num  |      100      |the max luminance in SDR image   | 
| b                             |  float num  |       0       |the min luminance in SDR image   | 
| alpha                         |  float num  |     1000      |the max luminance in HDR image   | 
| beta                          |  float num  |       0       |the min luminance in HDR image   |   

### 2.3 hdr_to_sdr
```
import cv2
import numpy as np
 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from hdr_to_sdr import SdrConversion

if __name__ == '__main__':
    SDRConvention = SdrConversion()
    image_path = r"../example-data/hdr.png"
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) / 65535.
    image_rgb = np.float32(image_bgr[..., ::-1].copy())
    image_rgb =  np.from_numpy(image_rgb)
    image_rgb_hdr_nolinear = SDRConvention.hdr_to_sdr(image_rgb)
    image_rgb_hdr_nolinear = image_rgb_hdr_nolinear.cpu().detach().numpy()
    image_bgr_hdr_nolinear = image_rgb_hdr_nolinear[..., ::-1]
    cv2.imwrite(r"../example-data/sdr_2.png", np.uint8(image_bgr_hdr_nolinear * 255.))
```



## Citations
More details can be found in [confluence](http://confluence.sm/pages/viewpage.action?pageId=16425244).

> Author:   season.cheng  
> Contact:  season.cheng@smartmore.com



## Update
   - July 27, 2021
        + init commit.
