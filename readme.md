# AutoGluon Installation Guide

AutoGluon is a user-friendly machine learning toolkit for automating machine learning tasks. This guide provides installation instructions for both CPU and GPU environments.

---

## Installation for CPU

### Step 1: Create a Conda Environment
```bash
conda create -n ag python=3.10
conda activate ag
```

### Step 2: Install Mamba
Mamba is a faster package manager for Conda.
```bash
conda install -c conda-forge mamba
```

### Step 3: Install AutoGluon
```bash
mamba install -c conda-forge autogluon
```

### Step 4: (Optional) Install Ray for Faster Training
Ray can optimize the training process.
```bash
mamba install -c conda-forge "ray-tune >=2.10.0,<2.32" "ray-default >=2.10.0,<2.32"
```

---

## Installation for GPU

### Step 1: Create a Conda Environment
```bash
conda create -n ag python=3.11
conda activate ag
```

### Step 2: Install Mamba
```bash
conda install -c conda-forge mamba
```

### Step 3: Install AutoGluon with GPU Support
Ensure that the GPU-specific version of PyTorch is installed.
```bash
mamba install -c conda-forge -c pytorch -c nvidia autogluon "pytorch=*=*cuda*"
```

### Step 4: (Optional) Install Ray for Faster Training
```bash
mamba install -c conda-forge "ray-tune >=2.10.0,<2.32" "ray-default >=2.10.0,<2.32"
```

---

## Verifying Installation

After installation, verify that AutoGluon and its dependencies are correctly installed by running the following commands:

```bash
python -c "import autogluon; print('AutoGluon version:', autogluon.__version__)"
python -c "import torch; print('Torch version:', torch.__version__)"
```

If you encounter any issues during installation, please ensure that your Conda environment is correctly set up and that your system meets the required specifications.

---

## Example: Starting AutoGluon Training

### Training Process Output
```plaintext
Beginning AutoGluon training...
AutoGluon will save models to 'd:\Mestrado\Aulas\AutogluonModels\ag-20241211_023735'
         Data  Turbidity        pH  Alkalinity  DissolvedOxygen  \
0  2018-01-01   7.367186  6.845486   25.788530         6.715897   
1  2018-01-02  35.755183  6.807069   19.943425         5.233228   
2  2018-01-03   9.341618  6.906234   19.910010         5.856647   
3  2018-01-04  16.105204  6.817615   20.249162         6.004393   
4  2018-01-05  16.531502  7.045985   18.217803         6.050563   

   RainAccumulation1D  RainAccumulation2D  RainAccumulation3D  \
0            0.001155            2.794847            2.318741   
1            0.001092            0.255247            0.718060   
2            0.012032            0.013785            0.029363   
3            0.002139            1.929273            0.788018   
4            0.002479            1.077661            0.135524   

   RainAccumulation4D  RainAccumulation5D  RainAccumulation6D  \
0            1.460683            9.169686           26.207817   
1            1.446020            2.584408           14.293565   
2            0.007139            0.389962            1.203419   
3            0.679701            1.694040            7.949185   
4            0.618780            5.573755            8.767472   

   RainAccumulation7D  RainAccumulation15D  RainAccumulation30D  RiverLevel  
0           19.561825           103.933015            52.559659   12.005540  
1           19.873974             7.830140            25.022983   11.948240  
2            4.060543            12.905984            -0.510383   11.890858  
3            9.757877            10.163177            52.038419   11.993244  
4           24.552537             2.447773            33.673292   12.046827  
=================== System Info ===================
AutoGluon Version:  1.2
Python Version:     3.11.11
Operating System:   Windows
Platform Machine:   AMD64
Platform Version:   10.0.19045
CPU Count:          12
GPU Count:          1
Memory Avail:       27.51 GB / 47.30 GB (58.2%)
Disk Space Avail:   97.79 GB / 931.50 GB (10.5%)
===================================================

Fitting with arguments:
{'enable_ensemble': True,
 'eval_metric': WQL,
 'hyperparameters': 'default',
 'known_covariates_names': ['pH',
                            'Alkalinity',
                            'RainAccumulation15D',
                            'RiverLevel'],
 'num_val_windows': 1,
 'prediction_length': 10,
 'quantile_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
 'random_seed': 123,
 'refit_every_n_windows': 1,
 'refit_full': False,
 'skip_model_selection': False,
 'target': 'Turbidity',
 'verbosity': 2}

Inferred time series frequency: 'D'
Provided train_data has 2373 rows, 1 time series. Median time series length is 2373 (min=2373, max=2373). 

Provided data contains following columns:
	target: 'Turbidity'
	known_covariates:
		categorical:        []
		continuous (float): ['pH', 'Alkalinity', 'RainAccumulation15D', 'RiverLevel']
	past_covariates:
		categorical:        []
		continuous (float): ['DissolvedOxygen', 'RainAccumulation1D', 'RainAccumulation2D', 'RainAccumulation3D', 'RainAccumulation4D', 'RainAccumulation5D', ...]

To learn how to fix incorrectly inferred types, please see documentation for TimeSeriesPredictor.fit

AutoGluon will gauge predictive performance using evaluation metric: 'WQL'
	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
===================================================

Starting training. Start time is 2024-12-10 23:37:37
Models that will be trained: ['SeasonalNaive', 'RecursiveTabular', 'DirectTabular', 'NPTS', 'DynamicOptimizedTheta', 'AutoETS', 'ChronosZeroShot[bolt_base]', 'ChronosFineTuned[bolt_small]', 'TemporalFusionTransformer', 'DeepAR', 'PatchTST', 'TiDE']
Training timeseries model SeasonalNaive. 
	-0.9473       = Validation score (-WQL)
	0.01    s     = Training runtime
	1.94    s     = Validation (prediction) runtime
Training timeseries model RecursiveTabular. 
	-1.0154       = Validation score (-WQL)
	1.26    s     = Training runtime
	0.23    s     = Validation (prediction) runtime
Training timeseries model DirectTabular. 
	-0.6225       = Validation score (-WQL)
	15.24   s     = Training runtime
	0.20    s     = Validation (prediction) runtime
Training timeseries model NPTS. 
	-0.6673       = Validation score (-WQL)
	0.01    s     = Training runtime
	1.56    s     = Validation (prediction) runtime
Training timeseries model DynamicOptimizedTheta. 
	-0.6983       = Validation score (-WQL)
	0.01    s     = Training runtime
	31.63   s     = Validation (prediction) runtime
Training timeseries model AutoETS. 
	-0.6739       = Validation score (-WQL)
	0.01    s     = Training runtime
	3.29    s     = Validation (prediction) runtime
Training timeseries model ChronosZeroShot[bolt_base]. 
	-0.6424       = Validation score (-WQL)
	3.04    s     = Training runtime
	2.47    s     = Validation (prediction) runtime
Training timeseries model ChronosFineTuned[bolt_small]. 
	Saving fine-tuned model to d:\Mestrado\Aulas\AutogluonModels\ag-20241211_023735\models\ChronosFineTuned[bolt_small]\W0\fine-tuned-ckpt
	-0.5924       = Validation score (-WQL)
	267.78  s     = Training runtime
	0.04    s     = Validation (prediction) runtime
Training timeseries model TemporalFusionTransformer. 
	-0.5773       = Validation score (-WQL)
	60.58   s     = Training runtime
	0.04    s     = Validation (prediction) runtime
Training timeseries model DeepAR. 
	-0.6294       = Validation score (-WQL)
	25.81   s     = Training runtime
	0.09    s     = Validation (prediction) runtime
Training timeseries model PatchTST. 
	-0.6157       = Validation score (-WQL)
	16.73   s     = Training runtime
	0.03    s     = Validation (prediction) runtime
Training timeseries model TiDE. 
	-0.6809       = Validation score (-WQL)
	67.55   s     = Training runtime
	0.03    s     = Validation (prediction) runtime
Fitting simple weighted ensemble.
	Ensemble weights: {'ChronosZeroShot[bolt_base]': 0.03, 'PatchTST': 0.54, 'TemporalFusionTransformer': 0.43}
	-0.5544       = Validation score (-WQL)
	0.77    s     = Training runtime
	2.53    s     = Validation (prediction) runtime
Training complete. Models trained: ['SeasonalNaive', 'RecursiveTabular', 'DirectTabular', 'NPTS', 'DynamicOptimizedTheta', 'AutoETS', 'ChronosZeroShot[bolt_base]', 'ChronosFineTuned[bolt_small]', 'TemporalFusionTransformer', 'DeepAR', 'PatchTST', 'TiDE', 'WeightedEnsemble']
Total runtime: 501.07 s
Best model: WeightedEnsemble
Best model score: -0.5544
```

---

## Additional Resources

- [AutoGluon Documentation](https://auto.gluon.ai)
- [Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [Mamba Documentation](https://mamba.readthedocs.io/)

---

Happy AutoML!

