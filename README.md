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
## EWC
python -u ./examples/main_ewc_market.py -b 128 --height 256 --width 128 --icm True --kd True --end-cm True -a resnet50 -d market1501 --wu 0.25 --momentum 0.1 --num-instances 16 --ci 0 --logs-dir ./result/market_ewc_icm_kd_ecm_ci0  --iters 100
python -u ./examples/main_ewc_market.py -b 128 --height 256 --width 128  -a resnet50 -d market1501 --w 0.25 --momentum 0.1 --num-instances 16 --wu 0.25 --ci 0 --logs-dir ./result/market_ewc_ci0  --iters 100

## MAS
python -u ./examples/main_mas_market.py -b 128 --height 256 --width 128 --icm True --kd True --end-cm True -a resnet50 -d market1501 --wu 0.25 --momentum 0.1 --num-instances 16 --ci 0 --logs-dir ./result/market_mas_icm_kd_ecm_ci0  --iters 100
python -u ./examples/main_mas_market.py -b 128 --height 256 --width 128  -a resnet50 -d market1501 --w 0.25 --momentum 0.1 --num-instances 16 --wu 0.25 --ci 0 --logs-dir ./result/market_mas_ci0  --iters 100


## SI
python -u ./examples/main_si_market.py -b 128 --height 256 --width 128 --icm True --kd True --end-cm True -a resnet50 -d market1501 --wu 0.25 --momentum 0.1 --num-instances 16 --ci 0 --logs-dir ./result/market_si_icm_kd_ecm_ci0  --iters 100
python -u ./examples/main_si_market.py -b 128 --height 256 --width 128  -a resnet50 -d market1501 --w 0.25 --momentum 0.1 --num-instances 16 --wu 0.25 --ci 0 --logs-dir ./result/market_si_ci0  --iters 100
```
