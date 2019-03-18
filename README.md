f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks
===================================================================

![Overview](GraphicalAbstract.pdf)

Code for reproducing **f-AnoGAN** training and anomaly scoring presented in [*"f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks"*](https://authors.elsevier.com/a/1Yexi4rfPly5Mm) ([accepted manuscript](https://github.com/tSchlegl/f-AnoGAN/tree/master/paper/fastAnoGAN.pdf)). This work extends **AnoGAN**: ["Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery"](https://rd.springer.com/chapter/10.1007/978-3-319-59050-9_12).


## Referencing and citing f-AnoGAN
If you use (parts of) this code in your work please refer to this citation:

```
Schlegl, T., Seeböck, P., Waldstein, S.M., Langs, G., Schmidt-Erfurth, U., 2019. f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks. Medical Image Analysis 54, 30-44. DOI: https://doi.org/10.1016/j.media.2019.01.010
```

## Prerequisites

- Python (2.7), TensorFlow (1.2), NumPy, SciPy, Matplotlib
- A recent NVIDIA GPU

## f-AnoGAN building blocks

- `wgangp_64x64.py`: Training a 64x64 WGAN architecture yields a trained generator (G) and discriminator (D). Modifies and extends (including `tflib/`) Ishaan Gulrajani's Tensorflow implementation of the WGAN-GP model proposed in ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028) ([GitHub](https://github.com/igul222/improved_wgan_training)).
- `z_encoding_izif.py`: Training the **izi_f** encoder (E) based on the trained WGAN model (G and D). Please refer to the full paper for more detailed information.
- `anomaly_detection.py`: Code for anomaly scoring utilizing the trained G, D, and E.

## Setting (image) paths

Image paths are set in `tflib/img_loader.py`. Images should be provided as "*.png" files structured in the following way:
- `trainset_path`     = "path-to-folder-holding-normal-training-images"
- `trainset_val_path` = "path-to-folder-holding-normal-validation-images"
- `test_normal_path`  = "path-to-folder-holding-normal-test-images"
- `test_anom_path`    = "path-to-folder-holding-anom-test-images"
Please edit that file to specify the paths to your datasets.


Results of related research work are provided at [CIR](http://www.cir.meduniwien.ac.at/research/anomalies/).