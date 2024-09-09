# cv
Computer Vision Projects
1. cifar.ipynb - Classification - Pytorch Tutorial 
    - Code source:  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    - Dataset:      torchvision.datasets.CIFAR10()
    - Model:        cnn
    - Classes:      'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
2. mnist.ipynb - Classification
    - Dataset:      https://www.kaggle.com/datasets/oddrationale/mnist-in-csv (.CSV-File)
    - Model:        cnn
    - Classes:      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
3. traffic_sign_detection.ipynb - Classification 
    - Dataset:      https://www.kaggle.com/datasets/pkdarabi/cardetection
    - Model:        cnn, cnn+more_filter+batch_normalization, cnn+dropout+more_fc_layer, resnet18_pretrained
    - Classes:      'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 
                    'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
                    'Speed Limit 80', 'Speed Limit 90', 'Stop', 'No Traffic Sign'
    - Disclaimer:   There is an extra kaggle notebook version that does not work with the current environment because it was run using Kaggle and its extra GPU power. (https://www.kaggle.com/code/nils123444/traffic-sign-detection-cnn-variations-resnet)
