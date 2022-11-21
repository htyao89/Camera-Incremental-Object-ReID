# Camera-Incremental-Object-ReID

### Installation

```shell
cd Camera-Incremental-Object-ReID
python setup.py develop
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets Market-1501, and MSMT17 from (https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/data.zip), which is provided by existing public code ClusterContrast(https://github.com/alibaba/cluster-contrast).
Then unzip them under the directory like
```
examples/data/market1501/Market-1501-v15.09.15
exmpales/data/veri/VeRi
```

## Training

```
python -u examples/main_ike.py --weight 0.25 -b 128 -a resnet50 -d market1501 --w 0.25 --momentum 0.1 --num-instances 16 --logs-dir ./result/market
```
