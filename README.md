# keras-object-detection-web-api

A simple wrapper over keras object detection libraries and provide web api using flask

# Usage

### Detect objects in an image using YOLO algorithm

The demo code below can be found in [keras_object_detection/demo/yolo_predict.py](keras_object_detection/demo/yolo_predict.py)

The demo codes takes in the image [keras_object_detection/demo/images/test.jpg](keras_object_detection/demo/images/test.jpg) and output the detected boxes with class labels

```python
import os

import scipy
from matplotlib.pyplot import imshow

from keras_object_detection.library.yolo import YoloObjectDetector
from keras_object_detection.library.yolo_utils import generate_colors, draw_boxes

model_dir_path = 'keras_object_detection/models'

image_file = 'keras_object_detection/demo/images/test.jpg'

detector = YoloObjectDetector()
detector.load_model(model_dir_path)

image, out_scores, out_boxes, out_classes = detector.predict

# Print predictions info
print('Found {} boxes for {}'.format(len(out_boxes), image_file))
# Generate colors for drawing bounding boxes.
colors = generate_colors(detector.class_names)
# Draw bounding boxes on the image file
draw_boxes(image, out_scores, out_boxes, out_classes, detector.class_names, colors)
# Save the predicted bounding box on the image
image.save(os.path.join("keras_object_detection/demo/out", image_file), quality=90)
output_image = scipy.misc.imread(os.path.join("out", image_file))
imshow(output_image)
```

Below is the image before detection:

![image-before](keras_object_detection/demo/images/test.jpg)

Here is the image after detection:

![image-after](keras_object_detection/demo/out/images/test.jpg)
