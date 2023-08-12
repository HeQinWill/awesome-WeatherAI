# Awesome Weather AI [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

<img src='./awesome-weatherai.jpg'>

This repository contains a curated list of papers, datasets, models, surveys, and other resources related to using AI for weather forecasting and climate modeling.  

It includes key papers that have advanced the state-of-the-art in areas like global forecasting, precipitation prediction, and climate projection using deep learning techniques. Popular benchmark datasets and open source implementations of influential models are also listed. The brief description is summarized by GPT.  

The goal of this repo is to provide a overview of the field to help researchers quickly get up to speed on the current progress and opportunities in applying AI to weather and climate domains. Contributions are welcome!  

## Contents

- [Papers](#papers)
- [Datasets](#datasets)
- [Models](#models)
- [Surveys](#surveys)
- [Blog Posts & News](#blog-posts--news)

## Papers

- [Accurate medium-range global weather forecasting with 3D neural networks](https://www.nature.com/articles/s41586-023-06185-3) - Kaifeng Bi et al., 2023

  Used 3D convolutional networks on a icosahedral mesh grid to generate medium-range global weather forecasts. Outperformed operational forecasts from ECMWF.

- [Can Machines Learn to Predict Weather? Using Deep Learning to Predict Gridded 500-hPa Geopotential Height From Historical Weather Data](https://onlinelibrary.wiley.com/doi/abs/10.1029/2019MS001705) - Jonathan A. Weyn et al., 2019

  Proposed a CNN model for weather forecasting basic atmospheric variables. Showed potential for ML in weather forecasting.

- [ClimateBench v1.0: A Benchmark for Data-Driven Climate Projections](https://onlinelibrary.wiley.com/doi/abs/10.1029/2021MS002954) - D. Watson-Parris et al., 2022

  Introduced a benchmark dataset and models for evaluating data-driven climate emulators.

- [ClimaX: A foundation model for weather and climate](http://arxiv.org/abs/2301.10343) - Tung Nguyen et al., 2023

  Proposed a Transformer-based model for general weather and climate prediction tasks. Showed strong performance with pretraining.

- [Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](http://arxiv.org/abs/2101.01000) - Haitao Lin et al., 2021

  Proposed a graph convolution model for meteorological forecasting that captures local spatial patterns.

- [Deep learning for twelve hour precipitation forecasts](https://www.nature.com/articles/s41467-022-32483-x) - Lasse Espeholt et al., 2022

  Showed a deep learning model that can effectively forecast precipitation and outperform operational models.

- [FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead](http://arxiv.org/abs/2304.02948) - Kang Chen et al., 2023

  Proposed an ensemble ML model for global medium-range forecasting that extends skillful forecasts to 10+ days.

- [FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](http://arxiv.org/abs/2202.11214) - Jaideep Pathak et al., 2022

  Introduced a Fourier neural operator model for global high-resolution weather forecasting.

- [FuXi: A cascade machine learning forecasting system for 15-day global weather forecast](http://arxiv.org/abs/2306.12873) - Lei Chen et al., 2023

  Proposed a cascaded ML model that provides skillful 15-day global forecasts.

- [GraphCast: Learning skillful medium-range global weather forecasting](http://arxiv.org/abs/2212.12794) - Remi Lam et al., 2022

  Showed a GNN model that surpasses operational medium-range weather forecasts.

- [Improving Data-Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere](https://onlinelibrary.wiley.com/doi/abs/10.1029/2020MS002109) - Jonathan A. Weyn et al., 2020

  Improved prior CNN weather forecasting model using cubed sphere mapping and other enhancements.

- [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) - Yuchen Zhang et al., 2023

  Proposed NowcastNet model for extreme precipitation nowcasting that outperforms operational models.

- [Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather Prediction Models](https://onlinelibrary.wiley.com/doi/abs/10.1029/2021MS002502) - Jonathan A. Weyn et al., 2021

  Showed a large DL model ensemble can provide skillful subseasonal forecasts.

- [SwinVRNN: A Data-Driven Ensemble Forecasting Model via Learned Distribution Perturbation](http://arxiv.org/abs/2205.13158) - Yuan Hu et al., 2023

  Proposed a stochastic weather forecasting model with learned distribution perturbation for ensemble forecasts.

## Datasets

- [WeatherBench](https://github.com/pangeo-data/WeatherBench) - Stephan Rasp et al., 2020

  A benchmark dataset derived from ERA5 for evaluating data-driven weather forecasting models.

- [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/index.html) - Stephan Rasp et al.

  Extended version of WeatherBench with more variables.

## Models

- [ClimaX](https://github.com/microsoft/ClimaX) - Microsoft, 2023

  Transformer model for weather and climate prediction.

- [FourCastNet](https://github.com/pangeo-data/ECKFNet) - Pangeo, 2022

  Fourier neural operator model for global weather forecasting.

- [GraphCast](https://github.com/DeepMind-Scholarship/graphcast) - DeepMind, 2022

  GNN model for medium-range weather forecasting.

- [ai-models](https://github.com/ecmwf-lab/ai-models) - ECMWF lab, 2023

  Implementations of ML weather models from ECMWF.

## Surveys

- [The rise of machine learning in weather forecasting](https://www.ecmwf.int/en/about/media-centre/science-blog/2023/rise-machine-learning-weather-forecasting) - ECMWF, 2023

  Discussion of recent advances in ML for weather forecasting from ECMWF scientists.

## Blog Posts & News

- [The AI Forecaster: Machine Learning Takes On Weather Prediction](http://eos.org/research-spotlights/the-ai-forecaster-machine-learning-takes-on-weather-prediction) - Eos, 2022

  News article on using ML for weather forecasting.
