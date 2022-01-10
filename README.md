# GamutConversion
This repository mainly contains gamut conversion methods including gamut-compress and gamut-expand,
which could be used in different circumstances.

## 1. **Introduction**
### 1.1 Gamut Conversion
#### 1.1.1 gamut
In color reproduction, including computer graphics and photography, the gamut is a certain complete subset of colors. 
The most common usage refers to the subset of colors which can be accurately represented in a given circumstance, such as within a given color space or by a certain output device.


Common color gamuts:
* bt.709 (Rec. 709) – ITU-R Recommendation for HDTV
* BT.2020 (Rec.2020) – ITU-R Recommendation for UHDTV
* Rec.2100 – ITU-R Recommendation for HDR-TV 
* Display P3 – Apple designed gamut
* Adobe RGB – 1998 Adobe standard


#### 1.1.2 gamut compress
Convert from large gamut to small gamut:

Take the process 'SDR BT.2020 ---> SDR BT.709' as an example, which could be probably the most common scenario of gammut conversion.


#### 1.1.3 gamut expand
Convert from small gamut to large gamut:


### 1.2 HDR -> SDR ( for gamut compress )  
*SDR*: Standard Dynamic Range content, which is publicly known as SDR, can be defined as a signal that is normally produced to be viewed
at a peak luminance of 100 cd/m2(nits) in the reference viewing environment, a black level close to zero and a bit-depth of 8 bits.   

*HDR*: As defined in Recommendation ITU-R BT.2100, High Dynamic Range Television(HDR-TV) provides viewers with an enhanced visual experience
by providing images that have been produced to look correct on brighter displays, that provide much brighter highlights, and that improved
details in dark areas.

The process of HDR -> SDR is presented as below:
![hdr_to_sdr.png](https://i.loli.net/2021/07/27/Z3Rc5pkuWVNqd7Y.png)

## 2. **Usage**   
### 2.1 prepare
```
   |--GamutConversion
      |--color
         |--color_model.py  # contains necessary colourspace conversion methods
         |--     :
         |--     :
      |--gamut_compress  # methods of compression
         |--closest_uvY.py
         |--closest_xyY.py
         |--twp_xyY.py
         |--      :
      |--gamut_expand   # methods of expand
         |--matrix_XYZ.py
         |--      :
         |--      :
      |--hdr_2_sdr   # procedure of HDR(bt.2020) -> SDR(bt.2020)
         |--hdr_to_sdr.py  # serves as the pre-steps of gamut compression(SDR bt.2020 -> SDR bt.709).
         |--      :
         |--      :
      |--example-data
         |--hdr.png
         |--   :
         |--   :
      hdr_to_sdr_example.py  # infer the sdr(bt.709) output.
      README.md
```

### 2.2 gamut compress
#### 2.2.1 Toward-White-Point + xyY colourspace



#### 2.2.2 Closest + xyY colourspace



#### 2.2.3 Closest + uvY colourspace



### 2.3 gamut expand

#### 2.3.1 Matrix conversion + XYZ colourspace


<!-- #### Parameter initial:  
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
 -->


## Citations
<!-- More details can be found in [confluence](http://confluence.sm/pages/viewpage.action?pageId=16425244). -->

> Author:   mengmeng.liu  
> Contact:  mengmeng.liu@smartmore.com


