## Communication-Efficient Diffusion Strategy for Performance Improvement of Federated Learning with Non-IID Data

## Official Implementation
![Overview](https://github.com/seyoungahn/JSAC_FedDif/blob/main/FedDif_overview.png "Overview of FedDif")

Paper: https://arxiv.org/abs/2207.07493

Seyoung Ahn, Soohyeong Kim, Yongseok Kwon, Joohan Park, Jiseung Youn, and Sunghyun Cho

***Abstract**─Federated learning (FL) is a novel learning paradigm that addresses the privacy leakage challenge of centralized learning. However, in FL, users with non-independent and identically distributed (non-IID) characteristics can deteriorate the performance of the global model. Specifically, the global model suffers from the weight divergence challenge owing to non-IID data. To address the aforementioned challenge, we propose a novel diffusion strategy of the machine learning (ML) model (FedDif) to maximize the FL performance with non-IID data. In FedDif, users spread local models to neighboring users over D2D communications. FedDif enables the local model to experience different distributions before parameter aggregation. Furthermore, we theoretically demonstrate that FedDif can circumvent the weight divergence challenge. On the theoretical basis, we propose the communication-efficient diffusion strategy of the ML model, which can determine the trade-off between the learning performance and communication cost based on auction theory. The performance evaluation results show that FedDif improves the test accuracy of the global model by 10.37% compared to the baseline FL with non-IID settings. Moreover, FedDif improves the number of consumed sub-frames by 1.28 to 2.85 folds to the latest methods except for the model compression scheme. FedDif also improves the number of transmitted models by 1.43 to 2.67 folds to the latest methods.*

## Installation
* All required packages are included in "requirements.txt"
* Installation commands (Python 3.8):
```
    conda create --name <your env name> --file requirements.txt
```

## Description
* Our experiments are all conducted by making the experiment codes in `main.py`
* Firstly set the system parameters and experiment name, and make `Experiment` object instance and call `FedDif()` function.
* Examples
  ```
  ## Exp. 3: Minimum tolerable QoS test
  for s_QoS_th in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
    params.alpha = 1.0
    params.epsilon = 0.0
    params.s_QoS_th = s_QoS_th
    experiment_name = 'FedDif_v4_cuda-{}_QoS-{}_cifar10_cnn'.format(int(args[1]), s_QoS_th)
    print("Experiment name: " + experiment_name)
    e = Experiment(experiment_name, params)
    e.FedDif()
    del e
  ```
* Two main parameters of FedDif to control the degree of diffusion
  1. `bar_delta` indicates the minimum tolerable IID distance $\varepsilon$ that prevents the over-diffusion for the minimization of IID distance. (Please refer to Fig. 4 of our paper)
  2. `s_QoS_th` indicates the minimum tolerable QoS $\gamma_{min}$ that prevents indiscriminate diffusion for considering the communication efficiency and outage probability. (Please refer to Fig. 5 of our paper)

* Other system parameters of our experiments
  1. `alpha` indicates the concentration parameter $\alpha$ that represents the degree of non-IID. It is equivalent to the concentration parameter of Dirichlet distribution. More detailed discription of the degree of non-IID is described in ["Hsu, Tzu-Ming Harry, Hang Qi, and Matthew Brown. "Measuring the effects of non-identical data distribution for federated visual classification." arXiv preprint arXiv:1909.06335 (2019)."](https://arxiv.org/abs/1909.06335).
  2. `t_model_version` indicates the type of ML model. We consider six ML models such as Logistic regression, CNN, FCN, SVM, LSTM, and ResNet34. Please set `t_model_version` variable as `"logistic"`, `"cnn"`, `"fcn"`, `"svm"`, `"lstm"`, and `"resnet34"`, respectively.
  3. `t_dataset_type` indicates the type of dataset. We consider four datasets such as CIFAR10, MNIST, FMNIST, and CIFAR100. Please set `t_dataset_type` variable as `"cifar10"`, `"mnist"`, `"fmnist"`, and `"cifar100"`, respectively.
  4. `t_init` indicates the weight initialization schemes. We consider two initialization schemes such as ["He"](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) and ["Xavier"](http://proceedings.mlr.press/v9/glorot10a) initialization. Please set `t_init` parameter as `he-uniform`, `he-normal`, `xavier-uniform`, and `xavier-normal`.

## Results
* All results of experiments are recorded in `save/<experiment name>_record1.csv`, `_record2.csv`, and `_record3.csv`. The schema of each file is as follows:
  1. `record1`: communication round | test accuracy | test loss | total diffusion rounds in the communication round
  2. `record2`: communcation round | diffusion round | diffusion efficiency | IID distances for all PUEs
  3. `record3`: communication round | diffusion round (`b`: global initialization, `a`: global aggregation) | # of bits sent | # of subframes sent
* Note that if there are no diffusion, record2 may not be created.
* For other results of FedDif, please refer to our paper.

## Citation
```
@article{ahn2022communication,
  title={Communication-Efficient Diffusion Strategy for Performance Improvement of Federated Learning with Non-IID Data},
  author={Ahn, Seyoung and Kim, Soohyeong and Kwon, Yongseok and Park, Joohan and Youn, Jiseung and Cho, Sunghyun},
  journal={arXiv preprint arXiv:2207.07493},
  year={2022}
}
```
