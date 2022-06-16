#!/bin/bash
# >>> conda initialize >>>
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

conda activate robot
work_dir=$(dirname $(realpath $0))
cd $work_dir
if [ ! -e "../capture/img" ];then
  echo "$work_dir/capture/img doen't exist, please enter manually img in labelme"
  labelme --autosave
else
  labelme --autosave ../capture/img
fi

