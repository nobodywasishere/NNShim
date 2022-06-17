#!/bin/sh

echo 'This builds everything needed for NNShim from a clean
source code pull of HotGauge (with NNShim support) on
a Linux machine'

set -e

rm -rf 3d-ice
./get_and_patch_3DICE.sh

cd 3d-ice
./install-superlu.sh
make all plugin
cd ..

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

git clone https://github.com/nobodywasishere/scale-sim-v2 scalesim

pip install -r scalesim/requirements.txt
pip install -r NNShim/requirements.txt
pip install -e scalesim/.
pip install -e HotGauge/.

echo 'Now just run `source venv/bin/activate` and you can start using NNShim by doing:'
echo '  cd NNShim'
echo '  ./NNShim.py -c default.toml'
