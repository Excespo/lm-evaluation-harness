#!/bin/bash

set -e

bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-geography-continue-all-data final nq_open_geography
bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-geography-continue-geography final nq_open_geography

bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-politics-continue-all-data final nq_open_politics
bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-politics-continue-politics final nq_open_politics

bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-literature-continue-all-data final nq_open_literature
bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-literature-continue-literature final nq_open_literature

bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-history-continue-all-data final nq_open_history
bash run/run_dense_domain.sh /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/fpm/Qwen2.5-Dense-1.5B-wiki-history-continue-history final nq_open_history
