#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${HOME}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate detectron2
work_dir=$(dirname $(realpath $0))
cd $work_dir
if [ ! -d "result/" ];then
  mkdir result
else
  rm -rf result
  mkdir result
fi
python train_net.py \
--train_dataset_names robot_cv_train  \
--train_json_paths ../annotation/labels/annotations.json \
--train_img_dirs ../annotation/labels \
--test_dataset_names wzh_test \
--test_json_paths ../annotation/labels/annotations.json \
--test_img_dirs ../annotation/labels \
--config-file config_file/R_50_1x.yaml \
--num-gpus 1 \
OUTPUT_DIR result

echo "+++++++++++++++++++"
echo "Finsh Train, result store BlendMask"
echo "+++++++++++++++++++"

if [ ! -d "Blendmask/" ];then
  mkdir Blendmask
else
  rm -rf Blendmask
  mkdir Blendmask
fi
cp result/{model_final.pth,config.yaml,metrics.json} Blendmask
cp ../annotation/labels/annotations.json Blendmask/train.json
exec /bin/bash
