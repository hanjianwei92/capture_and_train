conda activate robot
if [ ! $? -eq 0 ]; then
  conda init
  echo "Please restart script"
  sleep 5
  exit
fi
work_dir=$(dirname $(realpath $0))
cd $work_dir
if [ ! -e "../capture/img" ];then
  echo "$work_dir/capture/img doen't exist, please enter manually img in labelme"
  labelme --autosave
else
  labelme --autosave ../capture/img
fi
1
