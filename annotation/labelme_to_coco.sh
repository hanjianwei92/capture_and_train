conda activate robot
if [ ! $? -eq 0 ]; then
  conda init
  echo "Please restart script"
  sleep 5
  exit
fi

work_dir=$(dirname $(realpath $0))
cd $work_dir

if [ -d "labels/" ];then
  rm -rf labels
fi

python labelme2coco.py --input_dir=../capture/img

exec /bin/bash
