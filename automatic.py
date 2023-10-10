import cv2
import numpy as np
import time

# Video kaynağını başlatın (webcam kullanmak için 0'ı kullanabilirsiniz)
video_capture = cv2.VideoCapture(0)

# İlk kareyi alın
ret, previous_frame = video_capture.read()
previous_time = time.time()

while True:
    # Bir sonraki kareyi alın
    ret, current_frame = video_capture.read()

    # İki kare arasındaki farkı hesaplayın
    frame_diff = cv2.absdiff(previous_frame, current_frame)

    # Fark görüntüsünü eşikleyin
    thresholded_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    _, thresholded_diff = cv2.threshold(thresholded_diff, 30, 255, cv2.THRESH_BINARY)

    # Hareketli cisimleri mavi kare içinde gösterin ve konturları bulun
    moving_objects = current_frame.copy()
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(moving_objects, (x, y), (x + w, y + h), (255, 0, 0), 2)
            area = cv2.contourArea(contour)
            current_time = time.time()
            time_difference = current_time - previous_time
            previous_time = current_time
            if time_difference != 0:  # Sıfıra bölünmeyi önleyin
                speed = area / time_difference
            else:
                speed = 0
            cv2.putText(moving_objects, f"Area: {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(moving_objects, f"Speed: {speed:.2f} px/s", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

    # Hareketsiz cisimleri yeşil kare içinde gösterin
    stationary_objects = current_frame.copy()
    stationary_objects[np.where(thresholded_diff == 0)] = [0, 255, 0]

    # İki sonraki kareyi kaydedin
    previous_frame = current_frame

    # Pencereye görüntüleri gösterin
    cv2.imshow('Moving Objects (Blue)', moving_objects)
    cv2.imshow('Stationary Objects (Green)', stationary_objects)

    # Çıkış için "q" tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video yakalamayı serbest bırakın ve pencereleri kapatın
video_capture.release()
cv2.destroyAllWindows()
