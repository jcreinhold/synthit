# Quick Tutorial

## Install package

First download the package through git, i.e.,

`git clone https://github.com/jcreinhold/synthit.git`

If you are using conda, the easiest way to ensure you are up and running is to run the  `create_env.sh` script
located in the main directory. You run that as follows:

`. ./create_env.sh`

If you are *not* using conda, then you can try to install the package via the setup.py script, i.e.,
inside the `synthit` directory, run the following command:

`python setup.py install`

If you don't want to bother with any of this, you can create a Docker image or Singularity image via:

`docker pull jcreinhold/synthit`

or 

`singularity pull docker://jcreinhold/synthit`


## Polynomial Regression-based Synthesis

To do fast and simple synthesis for MR contrast synthesis, a reasonable go-to is polynomial regression. 
For instance, if we want to synthesize FLAIR images from T1-w images, we can do this with the following command:

```bash
synth-train -s train/t1/ -t train/flair/ -m train/masks/ -o pr_model.pkl -vv --n-samples 1e5 --ctx-radius 0 --patch-size 1 -r pr --poly-deg 2
```

The above command will fit a second degree polynomial to corresponding matching samples, using 100,000 (1e5) samples from
every image in the training set (i.e., all the images in the `train/t1/` directory). We can follow this with
prediction via the following command:

```bash
synth-predict -s test/t1/ -t pr_model.pkl -m test/masks/ -o results/ -vv
```

This command will synthesize FLAIR images from all the T1-w images in the `test/t1/` directory with the model created by the
previous command. The synthesized images will be saved in the `results/` directory with the same name as the original image
except the filename ends with `_syn.nii.gz`.
