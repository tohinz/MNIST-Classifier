# MNIST-Classifier
Simple classifier to classify MNIST images.

Requirements:
* Keras 2.1.4
* Tensorflow 1.5.0
* Numpy 1.14.1

To train the classifier on new data:
* `python mnist_classifier.py --train`

To view training statistics:
* `tensorboard --logdir log_dir/`

To predict images:
* `python mnist_classifier.py --predict --model path-to-model --img_path path-to-images`

Images should be stored in the following layout:
* class-0
    ** img1.jpg
    ** img2.jpg
** ...
* class-1
    ** img1.jpg
    ** img2.jpg
** ...
* ...
