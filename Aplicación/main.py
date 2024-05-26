from PIL import Image
import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2

model=YOLO(r"C:\Users\hrele\Documents\TESIS\experimentos\experimentos\yolov8\yolov8sx\weights\best.pt") 

with st.sidebar:
    st.title(':red[Deep Learning] para detección de :red[Equipo de Protección Personal] en imágenes en obras de construcción')
    st.divider()
    opcion=st.selectbox(label="Selecciona una opción", options=["Selecciona una opción", "Sube una imagen", "Sube un video", "Activar webcam", "Tomar fotografia"], index=0)
if opcion=="Sube una imagen":
    uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'jpeg', 'png'])
    if uploadFile is not None:
        image=Image.open(uploadFile)
        #st.image(image, width=100)
        image.save(uploadFile.name)
        results=model(uploadFile.name, save=False, save_frames=False, conf=0.7, iou=0.6)
        class_names = model.names
        keys = list(class_names.keys())
        values = list(class_names.values())
        for result in range(0,len(results)):
            image=results[result].orig_img
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            boxes=results[result].boxes.xyxy
            index_classes=results[result].boxes.cls
            class_names=results[result].names
            class_colors = {
                "helmet": (255, 0, 0),  # Red
                "no-helmet": (0, 255, 0),  # Green
                "no-vest": (0, 0, 255),  # Blue
                "person": (255, 255, 0),  # Yellow
                "vest": (255, 0, 255),  # Magenta
                # Add more class-color mappings as needed
                } 
            
            for box, index_class in zip(boxes, index_classes):
               # cls=int(box.cls[0])
                class_name=values[int(index_class)]
                color=class_colors.get(class_name, (255,255,255))
                x1,y1,x2,y2=box.tolist()
                cv2.rectangle(image,(int(x1), int(y1), int(x2), int(y2)), color, 2)
                cv2.putText(image, class_names[int(index_class)], (int(x1), int(y1) -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imwrite("output.jpg", image)
        st.image(image, width=600)
    else: 
        st.write("Sube imagen")
elif opcion=="Sube un video":
    uploadFile=st.file_uploader(label="Upload video", type=['mp4','mov'])
    temp_file_to_save = 'temp_file_1.mp4'
    # func to save BytesIO on a drive
    def write_bytesio_to_file(filename, bytesio):
        """
        Write the contents of the given BytesIO to a file.
        Creates the file or overwrites the file if it does
        not exist yet. 
        """
        with open(filename, "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(bytesio.getbuffer())
    
    class_names = model.names
    # Obtener todas las claves del diccionario como una lista
    keys = list(class_names.keys())
    # Obtener todas los valores del diccionario como una lista
    values = list(class_names.values()) 

    if uploadFile is not None:
        write_bytesio_to_file(temp_file_to_save, uploadFile)
        cap=cv2.VideoCapture(temp_file_to_save)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int 
        frame_placeholder=st.empty()
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            results=model(frame, stream=True, conf=0.5)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            for result in results:
                boxes=result.boxes
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    class_colors = {
                        "helmet": (255, 0, 0),  # Red
                        "no-helmet": (0, 255, 0),  # Green
                        "no-vest": (0, 0, 255),  # Blue
                        "person": (255, 255, 0),  # Yellow
                        "vest": (255, 0, 255),  # Magenta
                        # Add more class-color mappings as needed
                    } 
                    cls=int(box.cls[0])
                    class_name=values[cls]
                    color=class_colors.get(class_name, (255,255,255))
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 3)
                    cv2.putText(frame,class_name,(x1,y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    frame_placeholder.image(frame, width=600)
        cap.release()
elif opcion=="Activar webcam":
    class_names = model.names
    # Obtener todas las claves del diccionario como una lista
    keys = list(class_names.keys())
    # Obtener todas los valores del diccionario como una lista
    values = list(class_names.values())
    cap=cv2.VideoCapture(0)
    frame_placeholder=st.empty()
    stop_buttonpressed=st.button("Stop")
    while cap.isOpened() and not stop_buttonpressed:
        ret,frame=cap.read()
        if not ret:
            break
        results=model(frame, stream=True, conf=0.5)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        for result in results:
            boxes=result.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                class_colors = {
                    "helmet": (255, 0, 0),  # Red
                    "no-helmet": (0, 255, 0),  # Green
                    "no-vest": (0, 0, 255),  # Blue
                    "person": (255, 255, 0),  # Yellow
                    "vest": (255, 0, 255),  # Magenta
                    # Add more class-color mappings as needed
                } 
                cls=int(box.cls[0])
                class_name=values[cls]
                color=class_colors.get(class_name, (255,255,255))
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 3)
                cv2.putText(frame,class_name,(x1,y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                frame_placeholder.image(frame, width=600)
    cap.release()
elif opcion=="Tomar fotografia":
    class_names = model.names
    # Obtener todas las claves del diccionario como una lista
    keys = list(class_names.keys())
    # Obtener todas los valores del diccionario como una lista
    values = list(class_names.values())
    img_buffer=st.camera_input("Capturar")
    
    if img_buffer is not None:
        bytes_data = img_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        results=model(frame, stream=True, conf=0.5)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        for result in results:
            boxes=result.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                class_colors = {
                    "helmet": (255, 0, 0),  # Red
                    "no-helmet": (0, 255, 0),  # Green
                    "no-vest": (0, 0, 255),  # Blue
                    "person": (255, 255, 0),  # Yellow
                    "vest": (255, 0, 255),  # Magenta
                    # Add more class-color mappings as needed
                } 
                cls=int(box.cls[0])
                class_name=values[cls]
                color=class_colors.get(class_name, (255,255,255))
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 3)
                cv2.putText(frame,class_name,(x1,y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                st.image(frame, channels="RGB", width=600)   
