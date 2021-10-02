# CEBGAN and CBGAN

CEBGAN and CBGAN realized within the framework of Entropic GAN.

## Environment

* Python ≥ 3.7
* PyTorch ≥ 1.6.0
* Tensorboard ≥ 2.2.1
* Numpy ≥ 1.19.1
* Scipy ≥ 1.5.2

## Scripts

* **notebooks**: **Juperter Notebooks**
  * **MMD**: *Examine the MMD of different configurations of hyperparameters.*
  * **mode collapse**: Realization of §4.2 in the paper.
  * **multimodal regression**: Realization of §4.1 in the paper.
  * **surrogate likelihood**: Realization of §4.3.3 in the paper.
* **src**: **Python Scripts**
  * **train_cv_cbgan**: Run it to do cross validation for CBGAN.
  * **train_cv_cebgan**: Run it to do cross validation for CEBGAN.
  * **train_final_cbgan**: Run it to train the final CBGAN.
  * **train_final_cebgan**: Run it to train the final CEBGAN.
  * **generate_predictions_cbgan**: Run it to generate predictions using the final CBGAN.
  * **generate_predictions_cebgan**: Run it to generate predictions using the final CEBGAN.
  * **plot_comparison**: Compare groundtruth airfoils with their predictions.
  * **configs**
    * cegan.json: Hyperparameters of CEGAN.
    * cbegan.json: Hyperparameters of CEBGAN.
  * **models**
    * **layers**: Elemental PyTorch modules to be embedded in cmpnts.
      * BezierLayer generates airfoil data points.
    * **cmpnts**: PyTorch components for advanced applications.
      * Generators for a variety of applications.
      * Discriminators for a variety of applications.
    * **gans**: GAN containers built on top of each other.
    * **cgan**: Components of CEBGAN and CBGAN
    * **sinkhorn**: The Sinkhorn algorithm, which is modified from <https://github.com/jeanfeydy/global-divergences>
  * **utils**
    * **dataloader**: Data related tools for CEBGAN's training process.
      * AirfoilDataset that can generate samples with given parameters.
      * NoiseGenerator produces a given batch of uniform and normal noise combination in the latent space, and provide the probability density of each noise.
    * **metrics**: Where MMD is defined.
