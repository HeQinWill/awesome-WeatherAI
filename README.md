# Awesome Weather AI [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

<img src='https://github.com/HeQinWill/awesome-WeatherAI/blob/main/awesome-weatherai.jpg'>

This repository contains a curated list of papers, datasets, models, surveys, and other resources related to using AI for weather forecasting and climate modeling.  

It includes key papers that have advanced the state-of-the-art in areas like global forecasting, precipitation prediction, and climate projection using deep learning techniques. Popular benchmark datasets and open source implementations of influential models are also listed. The brief description is summarized by GPT.  

The goal of this repo is to provide an overview of the field to help researchers quickly get up to speed on the current progress and opportunities in applying AI to weather and climate domains. Contributions are welcome!  

## Contents

- [Papers](#papers)
- [Datasets](#datasets)
- [Models](#models)
    - [ClimaX](#climax)
    - [FourCastNet](#fourcastnet)
    - [FuXi](#fuxi)
    - [GraphCast](#graphcast)
    - [Pangu](#pangu)
    - [pySTEPS](#pysteps)
- [Blog Posts & News](#blog-posts--news)

## Papers

- [Accurate medium-range global weather forecasting with 3D neural networks](https://www.nature.com/articles/s41586-023-06185-3) - Kaifeng Bi et al., 2023 👉[Pangu](#pangu)

  Used 3D Earth-specific transformer (3DEST) on a icosahedral mesh grid to generate medium-range global weather forecasts. Outperformed operational forecasts from ECMWF.

- [Can Machines Learn to Predict Weather? Using Deep Learning to Predict Gridded 500-hPa Geopotential Height From Historical Weather Data](https://onlinelibrary.wiley.com/doi/abs/10.1029/2019MS001705) - Jonathan A. Weyn et al., 2019

  Proposed a CNN model for weather forecasting basic atmospheric variables. Showed potential for ML in weather forecasting.

- [ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling](http://arxiv.org/abs/2307.01909) - Tung Nguyen et al., 2023

  Designed an open-source PyTorch library to provide ML pipelines for climate and weather modeling.
  
- [ClimaX: A foundation model for weather and climate](http://arxiv.org/abs/2301.10343) - Tung Nguyen et al., 2023 👉[ClimaX](#climax)

  Proposed a Transformer-based model for general weather and climate prediction tasks. Showed strong performance with pretraining.

- [Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](http://arxiv.org/abs/2101.01000) - Haitao Lin et al., 2021

  Proposed a graph convolution model for meteorological forecasting that captures local spatial patterns.

- [Deep learning for twelve hour precipitation forecasts](https://www.nature.com/articles/s41467-022-32483-x) - Lasse Espeholt et al., 2022

  Showed a deep learning model that can effectively forecast precipitation and outperform operational models.

- [FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead](http://arxiv.org/abs/2304.02948) - Kang Chen et al., 2023

  Proposed an ensemble ML model for global medium-range forecasting that extends skillful forecasts to 10+ days.

- [FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](http://arxiv.org/abs/2202.11214) - Jaideep Pathak et al., 2022 👉[FourCastNet](#fourcastnet)

  Introduced a Fourier neural operator model for global high-resolution weather forecasting.

- [FuXi: A cascade machine learning forecasting system for 15-day global weather forecast](http://arxiv.org/abs/2306.12873) - Lei Chen et al., 2023 👉[FuXi](#fuxi)

  Proposed a cascaded ML model that provides skillful 15-day global forecasts.

- [GraphCast: Learning skillful medium-range global weather forecasting](http://arxiv.org/abs/2212.12794) - Remi Lam et al., 2022 👉[GraphCast](#graphcast)

  Showed a GNN model that surpasses operational medium-range weather forecasts.

- [Improving Data-Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere](https://onlinelibrary.wiley.com/doi/abs/10.1029/2020MS002109) - Jonathan A. Weyn et al., 2020

  Improved prior CNN weather forecasting model using cubed sphere mapping and other enhancements.

- [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) - Yuchen Zhang et al., 2023 👉[NowcastNet](#nowcastnet)

  Proposed NowcastNet model for extreme precipitation nowcasting that outperforms operational models.

- [Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather Prediction Models](https://onlinelibrary.wiley.com/doi/abs/10.1029/2021MS002502) - Jonathan A. Weyn et al., 2021

  Showed a large DL model ensemble can provide skillful subseasonal forecasts.

- [SwinVRNN: A Data-Driven Ensemble Forecasting Model via Learned Distribution Perturbation](http://arxiv.org/abs/2205.13158) - Yuan Hu et al., 2023

  Proposed a stochastic weather forecasting model with learned distribution perturbation for ensemble forecasts.
  
- [The Compatibility between the Pangu Weather Forecasting Model and Meteorological Operational Data](https://doi.org/10.48550/arXiv.2308.04460) - Wencong Cheng et al., 2023 👉[Pangu](#pangu)
  
  The Pangu-Weather model has been validated by the ECMWF and is compatible with various NWP systems, exhibiting stable forecasting capability and improved performance with better initial conditions.

- [YanTian: An Application Platform for AI Global Weather Forecasting Models](http://arxiv.org/abs/2410.04539) - Wencong Cheng et al., 2024

  Designed an application platform to enhance the usability of facilitate the AIGWFMs in operational weather forecasting.

- [Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling](https://arxiv.org/pdf/2309.15214) - Morteza Mardani et al., 2024 👉[CorrDiff](#corrdiff)

  Proposed a residual corrective diffusion modeling approach to enhance the accuracy and efficiency of kilometer-scale atmospheric downscaling. 

## Datasets

### ✨Raw archives
#### ERA5
- [hourly data on pressure levels from 1940](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)
- [hourly data on single levels from 1940](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)

#### ARCO-ERA5
[Analysis-Ready, Cloud Optimized ERA5 dataset](https://github.com/google-research/arco-era5)

- An ML-ready, unified (surface & atmospheric) version of the data in Zarr.  🔗`gs://gcp-public-data-arco-era5/ar/`
- A port of gaussian-gridded ERA5 data to Zarr.  🔗`gs://gcp-public-data-arco-era5/co/`
- All raw grib & NetCDF data.  🔗`gs://gcp-public-data-arco-era5/raw/`

Offers Zarr-formatted data in 0.25° lat-lon grids or native ERA5 grids, optimized for cloud access and AI-driven weather modeling.  

### ✨Benchmarks
- [WeatherBench](https://github.com/pangeo-data/WeatherBench)

  📒[paper](https://doi.org/10.1029/2020MS002203) 🔗[https](https://dataserv.ub.tum.de/index.php/s/m1524895) 🔗`ftp://m1524895:m1524895@dataserv.ub.tum.de`  
  A benchmark dataset derived from ERA5 for evaluating data-driven weather forecasting models.

- [WeatherBench 2](https://github.com/google-research/weatherbench2)

  📒[docs](https://weatherbench2.readthedocs.io/en/latest) 🔗`gs://weatherbench2/datasets`  
  Extended version of WeatherBench with more variables.

- [ClimateBench](https://github.com/duncanwp/ClimateBench)

  📒[paper](https://doi.org/10.1029/2021MS002954) 🔗[https](https://zenodo.org/record/7064308)  
  It consists of NorESM2 simulation outputs with associated forcing data processed in to a consistent format from a variety of experiments performed for CMIP6. Multiple ensemble members are included where available.  


## Models

### ✨Official implements

#### ClimaX 
- [ClimaX - Microsoft](https://github.com/microsoft/ClimaX)
- Universal Transformer model

#### FourCastNet
- [FourCastNet - NVIDIA](https://github.com/NVlabs/FourCastNet)
- Fourier neural operator model

#### CorrDiff
- [CorrDiff - NVIDIA](https://github.com/NVIDIA/modulus/tree/main/examples/generative/corrdiff)
- Residual Corrective Diffusion model for downscaling

#### FuXi
- [FuXi - Fudan](https://github.com/tpys/FuXi)
- Cascaded ML weather forecasting system

#### GraphCast
- [GraphCast - DeepMind](https://github.com/deepmind/graphcast)
- GNN model

#### NowcastNet
- [NowcastNet - Tsinghua](https://doi.org/10.24433/CO.0832447.v1)
- Nonlinear nowcasting model for extreme precipitation

#### Pangu
- [Pangu - Huawei](https://github.com/198808xc/Pangu-Weather)
- Earth-specific transformer (3DEST)

#### pySTEPS
- [pySTEPS](https://github.com/pySTEPS/pysteps)

### ✨Other kits

- [ai-models - ECMWF](https://github.com/ecmwf-lab/ai-models)  
  👉[FourCastNet](#fourcastnet) 👉[Pangu](#pangu)

- [OpenCastKit - High-Flyer](https://github.com/HFAiLab/OpenCastKit)  
  👉[FourCastNet](#fourcastnet) 👉[GraphCast](#graphcast)


## Blog Posts & News
- [The AI Forecaster: Machine Learning Takes On Weather Prediction](http://eos.org/research-spotlights/the-ai-forecaster-machine-learning-takes-on-weather-prediction) - Eos, 2022
- [The rise of machine learning in weather forecasting](https://www.ecmwf.int/en/about/media-centre/science-blog/2023/rise-machine-learning-weather-forecasting) - ECMWF, 2023
- [AI weatherman: the DeepMind researcher making faster, more accurate forecasts](https://www.nature.com/articles/d41586-024-03898-x) - Nature, 2024
