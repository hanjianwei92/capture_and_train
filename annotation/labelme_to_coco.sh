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

if [ -d "aug/" ];then
  rm -rf aug
fi

mkdir "aug"
python augment2.py --img_dir="../capture/img" --img_out_dir="aug" --labelme_dir="../capture/img" --labelme_dump_dir="aug"
python labelme2coco.py --input_dir="aug"

exec /bin/bash
