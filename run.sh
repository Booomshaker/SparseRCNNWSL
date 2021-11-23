#! /bin/bash
python3 projects/SparseRCNN/train_net.py \
--num-gpus 8 \
--config-file projects/SparseRCNN/configs/sparsercnn.wsddn.vgg16.yaml \
--dist-url 'tcp://127.0.0.1:50125' \
OUTPUT_DIR output/sparsercnnwsl_vgg16_voc2007_`date +'%Y-%m-%d_%H-%M-%S'`
