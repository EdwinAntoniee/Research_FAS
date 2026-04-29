import cv2
import time
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# --- 1. Inisialisasi Model AI ---
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(320, 320)) # det_size 320x320 agar FPS lebih ringan

# Ganti dengan nama file pickle yang kamu gunakan (misal: absensi_5.pkl)
DATABASE_FILE = "model_recognition_facenet.pkl" 
COSINE_THRESHOLD = 0.6552

print(f"Membaca database wajah dari {DATABASE_FILE}...")
try:
    with open(DATABASE_FILE, "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    print(f" [V] Berhasil memuat {len(known_face_names)} data wajah.")
except FileNotFoundError:
    print(f" [X] Error: File {DATABASE_FILE} tidak ditemukan!")
    exit()

# --- 2. Menyalakan Kamera (Gaya Veriface) ---
cap = None
# Mencoba beberapa indeks termasuk DirectShow untuk Windows
for index in [0, 1, 0 + cv2.CAP_DSHOW]:
    temp_cap = cv2.VideoCapture(index)
    time.sleep(1) # Memberikan waktu bagi hardware untuk bersiap
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"Kamera berhasil dibuka pada indeks {index}.")
        break
    temp_cap.release()

if cap is None:
    print("ERROR FATAL: Tidak dapat membuka kamera. Pastikan kamera terpasang dan driver sudah benar.")
    exit()

# --- 3. Proses Real-Time Tracking & Recognition ---
FRAME_SKIP = 3 # Diubah ke 3 agar video lanncar dan tidak nge-lag
frame_count = 0
saved_faces_info = [] # Menyimpan hasil deteksi agar kotak tidak hilang saat frame di-skip

print("\n--- Sistem Face Recognition Aktif (Tekan 'q' untuk keluar) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Logika eksekusi AI berdasarkan frame selang-seling
    if frame_count % FRAME_SKIP == 0:
        
        # Deteksi wajah menggunakan InsightFace
        faces = app.get(frame)
        
        temp_faces_info = []
        for face in faces:
            current_embedding = face.embedding
            name = "Tidak Dikenal"
            min_distance = 1.0

            # Bandingkan dengan Database (Pengenalan Identitas)
            for i, known_encoding in enumerate(known_face_encodings):
                distance = cosine(current_embedding, known_encoding)
                if distance < min_distance:
                    min_distance = distance
                    if min_distance < COSINE_THRESHOLD:
                        name = known_face_names[i]
            
            # Simpan data sementara
            temp_faces_info.append({
                "bbox": face.bbox.astype(np.int32),
                "name": name,
                "distance": min_distance
            })
        
        # Perbarui wajah yang akan digambar di layar
        saved_faces_info = temp_faces_info

    # Gambar Kotak dan Label dari data yang tersimpan (dilakukan di setiap frame)
    for info in saved_faces_info:
        bbox = info["bbox"]
        name = info["name"]
        min_distance = info["distance"]

        # Logika warna kotak: Hijau jika dikenali, Merah jika tidak
        if name != "Tidak Dikenal":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.rectangle(frame, (bbox[0], bbox[3] - 30), (bbox[2], bbox[3]), color, cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        text_label = f"{name} ({min_distance:.2f})"
        cv2.putText(frame, text_label, (bbox[0] + 6, bbox[3] - 6), font, 0.7, (255, 255, 255), 1)

    # Tampilkan Hasil
    cv2.imshow('Sistem Absensi Real-Time', frame)

    # Tombol Keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
print("\nSistem Dihentikan.")