# AITM-torch

Support Python and C++ 

- [Official Tensorflow implementation](https://github.com/xidongbo/AITM)
- [Paper](https://arxiv.org/abs/2105.08489)



## Dataset

Download and preprocess dataset use [script](https://github.com/xidongbo/AITM/blob/main/process_public_dataset.py) in [official Tensorflow implementation](https://github.com/xidongbo/AITM)
```python
python process_public_dataset.py

```

## Train && Test

```
cd python
mkdir out
python train.py
```

## AUC performance

Test AUC on [Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408) dataset

```
Test Resutt: click AUC: 0.6189267022220789 conversion AUC:0.6544229866061039
```

## Deployment
Pytorch c++ front is 2x faster than python version and 6x faster than TorchScript when inference.
We Implement a C++ version that which can load tochscript format file
There is a demo at [cpp](./cpp)

### build
Download libtorh [libtorch installing docs](https://pytorch.org/cppdocs/installing.html)

```
cd cpp
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```
