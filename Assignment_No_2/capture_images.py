import cv2
import os

# Folder where images will be saved
save_dir = "dataset/positive"
os.makedirs(save_dir, exist_ok=True)

url = "http://10.25.7.187:8080/video"

# Open webcam
cap = cv2.VideoCapture(url)
count = 0
max_images = 20  

print("[INFO] Press 's' to save the current frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image.")
        break

    # Display instructions on the frame
    cv2.putText(frame, f"Images saved: {count}/{max_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capture Positive Images", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save image
    if key == ord('s'):
        file_path = os.path.join(save_dir, f"pos_{count+1}.jpg")
        cv2.imwrite(file_path, frame)
        count += 1
        print(f"[INFO] Saved {file_path}")

        if count >= max_images:
            print("[INFO] Finished capturing images.")
            break

    # Press 'q' to quit without finishing
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
