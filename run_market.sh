## EWC
#python -u ./examples/main_ewc_market.py -b 128 --height 256 --width 128 --icm True --kd True --end-cm True -a resnet50 -d market1501 --wu 0.25 --momentum 0.1 --num-instances 16 --ci 0 --logs-dir ./result/market_ewc_icm_kd_ecm_ci0  --iters 100
python -u ./examples/main_ewc_market.py -b 128 --height 256 --width 128  -a resnet50 -d market1501 --w 0.25 --momentum 0.1 --num-instances 16 --wu 0.25 --ci 0 --logs-dir ./result/market_ewc_ci0  --iters 100

## MAS
#python -u ./examples/main_mas_market.py -b 128 --height 256 --width 128 --icm True --kd True --end-cm True -a resnet50 -d market1501 --wu 0.25 --momentum 0.1 --num-instances 16 --ci 0 --logs-dir ./result/market_mas_icm_kd_ecm_ci0  --iters 100
#python -u ./examples/main_mas_market.py -b 128 --height 256 --width 128  -a resnet50 -d market1501 --w 0.25 --momentum 0.1 --num-instances 16 --wu 0.25 --ci 0 --logs-dir ./result/market_mas_ci0  --iters 100


## SI
#python -u ./examples/main_si_market.py -b 128 --height 256 --width 128 --icm True --kd True --end-cm True -a resnet50 -d market1501 --wu 0.25 --momentum 0.1 --num-instances 16 --ci 0 --logs-dir ./result/market_si_icm_kd_ecm_ci0  --iters 100
#python -u ./examples/main_si_market.py -b 128 --height 256 --width 128  -a resnet50 -d market1501 --w 0.25 --momentum 0.1 --num-instances 16 --wu 0.25 --ci 0 --logs-dir ./result/market_si_ci0  --iters 100


