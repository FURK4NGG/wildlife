# wildlife

sudo apt update
sudo apt full-upgrade -y

sudo apt install --reinstall \
    python3-numpy \
    python3-simplejpeg \
    python3-picamera2 \
    python3-gpiozero




Camera  
  ↓  
Picamera2  
  ↓  
Hailo-8  
  ↓  
yolov11l.hef  
  ↓  
Hailo NMS  
  ↓  
Bounding boxes  
  ↓  
OpenCV  


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
