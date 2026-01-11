from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np

def coord_to_pixel(pointLon, pointLat, im, x_offset=0, y_offset=0):
    imageWidth, imageHeight = im.size
    # NYC bounding box coordinates
    ImageExtentLeft = -74.2591  # Western edge
    ImageExtentRight = -73.7004  # Eastern edge
    ImageExtentTop = 40.9176    # Northern edge
    ImageExtentBottom = 40.4774  # Southern edge
    
    x = imageWidth * ( pointLon - ImageExtentLeft ) / (ImageExtentRight - ImageExtentLeft)
    y = imageHeight * ( 1 - ( pointLat - ImageExtentBottom) / (ImageExtentTop - ImageExtentBottom))
    return x+x_offset, y+y_offset


im_path = "../data/nyc_map.png"  # Update with actual NYC map image if available
geo_coord_center = -73.9712, 40.7831  # NYC center (Times Square)

im = Image.open(im_path)
size = im.size
pic_x = size[0]/4
pic_y = size[1]/4

plt.imshow(im)
plt.scatter(pic_x, pic_y, c='red', s=20)


x, y = coord_to_pixel(geo_coord_center[0], geo_coord_center[1], im)
# print(pic_x-x, pic_y-y)
plt.scatter(x, y, c='blue', s=20, marker='x')

# united center
# x, y = coord_to_pixel(-87.67421094151734, 41.88068289515608, im)
# plt.scatter(x, y, c='green', s=20) 



plt.show()
