# AR-skyline

## Ideas
### extraction from GIS
- Different zoom for tiles at different distance from viewpoint.
- Account for curvature of  earth
- Radius in merters instead of degrees

## Goals
### Skyline extraction from image
- Robustness against clouds
- Different colors of sky, blue, grey, warm

## Previous works
### Skyline extraction from image
[Skyline Matching: A robust registration method between Video and GIS](https://www.researchgate.net/publication/245031401_Skyline_Matching_A_robust_registration_method_between_Video_and_GIS)
first using a **closing morpholocical** operation on the image to reduce artefacts
caused by thin structures such as: telegraph poles, wires and road lights. Then use **adaptive threshholding** to separate sky from image. (maybe order of these operation are wrong)