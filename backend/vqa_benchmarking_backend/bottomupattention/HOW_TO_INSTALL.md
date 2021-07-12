## Install instructions Detectron2 Based Feature Extractor

* Create a conda environment with **python 3.8**. 3.9 Does not work, because ray is a requirement that doesn’t have a wheel for 3.9 yet. Compiling it myself, it yields a lot of errors, some of them were not fixable.

`git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch`
`cd bottom-up-attention.pytorch`

* install Requirements
* `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch`
* `pip install opencv-python` 
* `pip install 'ray[default]'`

* we need to install a pre-build detectron wheel, because of library mismatches
`rm -rf detectron2`
`python -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html` 

* Compile a Gcc version < 10.1
* Set environment variables to your CC and CXX 
* `CC=/mount/arbeitsdaten/asr-2/vaethdk/imgfeats_final/gcc/dist/usr/local/bin/gcc`
* `CXX=/mount/arbeitsdaten/asr-2/vaethdk/imgfeats_final/gcc/dist/usr/local/bin/g++`  

* Install apex
	* `git clone https://github.com/NVIDIA/apex.git`
	* `cd apex`
	* `python setup.py install`
	* `cd ..`

* In setup.py, add include paths to include_dirs:
	* custom gcc includes
		* `/mount/arbeitsdaten/asr-2/vaethdk/imgfeats_final/gcc/dist/usr/local/include/c++/9.3.0`
	* custom thrust includes
		* `/mount/arbeitsdaten/asr-2/vaethdk/imgfeats_final/thrust`
* Then, build library
`python setup.py build develop`

* Download the model file, and put into main folder
…
* Change the defaults of the extract_features.py file accordingly
…

* Test
```
python3 extract_features.py —-mode caffe —-num-cpus 16 —-gpus '7'  —-extract-mode roi_feats -—min-max-boxes '10,100' —-config-file configs/bua-caffe/extract-bua-caffe-r101.yaml —-image-dir ./demo —-bbox-dir ./output -—out-dir ./output
```