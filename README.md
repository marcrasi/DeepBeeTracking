# DeepBeeTracking

## UNet

Train the UNet with `swift run TrainUNet`. This saves weights in "checkpoints/".

Do inference using this [Colab notebook][unet_nb] (which imports DeepBeeTracking). It
is configured to load some pretrained weights from the internet. You can edit
it to point at your own weights.

[unet_nb]: https://colab.research.google.com/github/marcrasi/DeepBeeTracking/blob/master/Notebooks/Bee_UNet.ipynb
