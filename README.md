# Training Generative Adversarial Networks with Limited Data :- StyleGAN ADA (Adaptive Discriminator Augmentation)

## Abstract :- 
Training generative adversarial networks (GAN) using too little data typically leads
to discriminator overfitting, causing training to diverge. We propose an adaptive
discriminator augmentation mechanism that significantly stabilizes training in
limited data regimes. The approach does not require changes to loss functions
or network architectures, and is applicable both when training from scratch and
when fine-tuning an existing GAN on another dataset. We demonstrate, on several
datasets, that good results are now possible using only a few thousand training
images, often matching StyleGAN2 results with an order of magnitude fewer
images. We expect this to open up new application domains for GANs. We also
find that the widely used CIFAR-10 is, in fact, a limited data benchmark, and
improve the record FID from 5.59 to 2.42.


## Architecture :- Architecture is same as StyleGAN2 but instead of directly feeding image to Discriminator we are augmenting to avoid overfitting in Discriminator .
So that we can train out StyleGAN with the Limited Data

If the Dataset size is around 1k then just finetune your model using some pretrained checkpoints . 


```
@misc{karras2020training,
      title={Training Generative Adversarial Networks with Limited Data}, 
      author={Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
      year={2020},
      eprint={2006.06676},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
