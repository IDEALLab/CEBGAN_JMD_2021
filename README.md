# CBGAN and CEBGAN for 2D Airfoil Inverse Design
CEBGAN and airfoil optimization code associated with our accepted JMD 2021 paper: "Inverse Design of 2D Airfoils using Conditional Generative Models and Surrogate Log-Likelihoods."

![Alt text](/cbgan_architecture.svg)
<!-- ![Alt text](/architecture2.svg) -->

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

<!-- Wei Chen, Kevin Chiu, and Mark Fuge. "Airfoil Design Parameterization and Optimization using Bézier Generative Adversarial Networks." AIAA Journal (2020) Accepted.

    @article{chen2020airfoil,
	  title={Airfoil Design Parameterization and Optimization using Bézier Generative Adversarial Networks},
	  author={Chen, Wei and Chiu, Kevin and Fuge, Mark},
	  journal={AIAA Journal},
	  volume={},
	  number={},
	  pages={},
	  year={2020},
	  publisher={American Institute of Aeronautics and Astronautics}
        } -->

## Required packages

<!-- - tensorflow<2.0.0
- pyDOE
- sklearn
- numpy
- matplotlib
- autograd -->

## Dataset

Our airfoil designs are optimized airfoils in SU2. The original airfoils are generated by [BezierGAN](https://github.com/IDEALLab/bezier-gan). BezierGAN was trained on [UIUC airfoil coordinates database](http://m-selig.ae.illinois.edu/ads/coord_database.html).

[//]: <The raw data contains variable number of points along airfoil curves. We created the training data by applying [B-spline interpolation](https://github.com/IDEALLab/airfoil-interpolation) on these designs and removed outlier designs.>

## CFD solver and airfoil optimization

We use [SU2](https://su2code.github.io/) as the CFD solver to evaluate the performance of the airfoil design.

## Usage

### Train/evaluate Conditional-Bézier-GAN

<!-- Go to Conditional-Bézier-GAN's directory:

```bash
cd conditional-bezier-gan
```

Train a Bézier-GAN or evaluate a trained Bézier-GAN:

```bash
python train_gan.py
```

positional arguments:
    
```
mode	train or evaluate
noise	noise dimension
```

optional arguments:

```
-h, --help            	show this help message and exit
--model_id		model ID
--save_interval 	number of intervals for saving the trained model and plotting results
```

The trained model and synthesized shape plots will be saved under directory `conditional-bezier-gan/trained_gan/<latent>_<noise>/<model_id>`, where `<latent>`, `<noise>`, and `<model_id>` are latent dimension, noise dimension, and model ID specified in the above arguments. -->

## Results

### Airfoil samples of training data:

<img src=sample_train.svg width="640">

### Quantitative performance of conditional GANs:

Reduction in instantaneous optimality gap:

<img src=graph1_merged_250.svg width="640">

Reduction in cumulative optimality gap:

<img src=graph3_merged_250.svg width="640">

Reduction in cumulative optimality gap of an example airfoil:

<img src=cbegan_graph2.svg width="640">

<!-- Randomly generated airfoils:

<img src=synthesized_noise.svg width="640"> -->

### Practicability of the Surrogate Log-Likelihood


