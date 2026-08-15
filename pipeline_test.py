import cv2
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
video_path = "video_h264.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Yazıyı tamamen silmek için kırpma miktarını artırdık
kirpma_miktari = 150
yeni_height = height - kirpma_miktari

# Kareleri kaydedeceğimiz klasörü oluştur
if not os.path.exists("keyframes"):
    os.makedirs("keyframes")

frame_count = 0
saniye = 0

print("Keyframe'ler çıkarılıyor...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    kirpilmis_kare = frame[0:yeni_height, :]
    
    # Saniyede sadece 1 kare (Keyframe) al (Sliding window için hazırlık)
    if frame_count % fps == 0:
        # Sadece insan(0) ve kamyon/forklift(7) sınıflarını filtrele
        results = model(kirpilmis_kare, classes=[0, 7], verbose=False)
        
        # Eğer ekranda takip ettiğimiz nesnelerden biri varsa bu kareyi kaydet
        if len(results[0].boxes) > 0:
            zaman_damgasi = f"00_{saniye:02d}"
            dosya_adi = f"keyframes/frame_{zaman_damgasi}.jpg"
            
            # Kareyi çizimsiz, ham haliyle modele göndermek için kaydet 
            # (Çizimli halini modelin kafasını karıştırmaması için kullanmıyoruz)
            cv2.imwrite(dosya_adi, kirpilmis_kare)
        saniye += 1
        
    frame_count += 1

cap.release()
print("İşlem tamamlandı! 'keyframes' klasörünü kontrol et.")