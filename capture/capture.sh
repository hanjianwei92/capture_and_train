conda activate robot
if [ ! $? -eq 0 ]; then
  conda init
  echo "Please restart script"
  sleep 5
  exit
fi
work_dir=$(dirname $(realpath $0))
cd $work_dir
echo $(python --version)
python take_photos.py
exec /bin/bash
