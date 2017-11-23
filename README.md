# SRResNet-AC-DRAGAN
DRAGAN Implementation with Auxiliary Classifier based on a modification from SRResNet

Attempt at recreating results from: Towards the Automatic Anime Characters Creation with Generative Adversarial Networks (https://arxiv.org/pdf/1708.05509.pdf)

Experimented with the CelebA Dataset.

TODO:
－ Add results on tagged anime dataset
－ Double check training procedure

Pretrained model available at https://drive.google.com/open?id=1JTpjlfVO-Frjr9_rdytxiuQZU7ixCF53

Download Cropped, Aligned CelabA Dataset at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Code is based on the following repositories:

- https://github.com/ai-tor/PyTorch-SRGAN
- https://github.com/jfsantos/dragan-pytorch
- https://github.com/pytorch/examples/blob/master/imagenet/main.py

To run the code, adjust `settings.py` and call `python3 main.py`.

## Results
Epoch 0-18:
<div>
    <img src='samples/fake_samples_epoch_000.png', width="48%">
    <img src='samples/fake_samples_epoch_001.png', width="48%">
    <img src='samples/fake_samples_epoch_002.png', width="48%">
    <img src='samples/fake_samples_epoch_003.png', width="48%">
    <img src='samples/fake_samples_epoch_004.png', width="48%">
    <img src='samples/fake_samples_epoch_005.png', width="48%">
    <img src='samples/fake_samples_epoch_006.png', width="48%">
    <img src='samples/fake_samples_epoch_007.png', width="48%">
    <img src='samples/fake_samples_epoch_008.png', width="48%">
    <img src='samples/fake_samples_epoch_009.png', width="48%">
    <img src='samples/fake_samples_epoch_010.png', width="48%">
    <img src='samples/fake_samples_epoch_011.png', width="48%">
    <img src='samples/fake_samples_epoch_012.png', width="48%">
    <img src='samples/fake_samples_epoch_013.png', width="48%">
    <img src='samples/fake_samples_epoch_014.png', width="48%">
    <img src='samples/fake_samples_epoch_015.png', width="48%">
    <img src='samples/fake_samples_epoch_016.png', width="48%">
    <img src='samples/fake_samples_epoch_017.png', width="48%">
    <img src='samples/fake_samples_epoch_018.png', width="48%">
</div>
