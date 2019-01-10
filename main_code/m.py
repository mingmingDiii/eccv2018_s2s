import scipy.misc as misc
from PIL import Image
import numpy as np
a = Image.open('/mnt/sda1/don/documents/project_paper/video_seg/data/youtube_vos/valid/Annotations/0a49f5265b/00000.png')
a = np.array(a)
print(a)