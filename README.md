# wildlife

```
sudo apt update
sudo apt full-upgrade -y

sudo apt install --reinstall \
    python3-numpy \
    python3-simplejpeg \
    python3-picamera2 \
    python3-gpiozero
```

cd ~/Desktop/wildlife_YOLO  

rm -rf .venv  

python3 -m venv --system-site-packages .venv  
source .venv/bin/activate  

python3 -c "from picamera2 import Picamera2; print('Picamera2 OK')"  
>Picamera2 OK  

python3 -c "import numpy; print('NumPy:', numpy.__version__)"  
>NumPy: 1.24.2  

python3 -c "from gpiozero import MotionSensor; print('gpiozero OK')"  
>gpiozero OK  

python3 -c "from hailo_platform import HEF, VDevice; print('HailoRT Python OK')"  
>hailortcli fw-control identify  

hailortcli parse-hef ~/Desktop/wildlife_YOLO/yolov11l.hef  
>Architecture: HAILO8
>Input: 640x640x3 UINT8
>Classes: 7
>Hailo NMS BY CLASS
>Score threshold: 0.200
>IoU threshold: 0.70



cat ~/Desktop/wildlife_YOLO/labels.txt  
0 = bear  
1 = boar  
2 = deer  
3 = wolf  
4 = cow  
5 = person  
6 = dog  

RUN  
cd ~/Desktop/wildlife_YOLO  
source .venv/bin/activate  
python3 wildlife_detector_hailo.py  


Picamera2  
    ↓  
Camera  
    ↓  
Hailo-8  
    ↓  
yolov11l.hef  
    ↓  
Hailo NMS  
    ↓  
Bounding boxes  
    ↓  
wildlife_detector_hailo.py  
    ↓  
PIR / Continuous  
    ↓  
Image / Video  


config.json  
hailort.log(otomatik olusturulur)  
labels.txt  
wildlife_detector.py  
yolov11l.hef  
outputs/(otomatik olusturulu)  


Features  
PIR modu  
Continuous modu  
Bear / Boar / Deer / Wolf tetikleme  
Fotoğraf  
Video  
Video süresi  
FPS min/max  
FPS overlay  
Raspberry Pi tarih/saat overlay  
Class'a özel box renkleri  
Default yeşil  
Aynı dosyayı birden fazla class klasörüne koyma  
outputs/bear/image  
outputs/bear/video  
Canlı kamera preview  
JSON'dan ayar yönetimi  




Hailo-8              ✅  
HailoRT 4.20.0       ✅  
yolov11l.hef         ✅  
640x640x3            ✅  
7 classes            ✅  
Hailo NMS            ✅  
labels.txt           ✅  
Picamera2            ✅  
