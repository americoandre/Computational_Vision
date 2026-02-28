import os
# Desativa a internet para não travar no erro de DNS
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

from ultralytics import YOLO

def iniciar_treino():
    # 1. Carrega o modelo de classificação que você baixou
    model = YOLO(r"weights/yolov8n-cls.pt")

    # 2. Caminho absoluto do dataset (use barras normais / )
    dataset_path = "C:/Users/Américo Umba André/Desktop/LIXO INDUSTRIAL/app/backend/dataset"

    # 3. Iniciar o treino
    model.train(
        data=dataset_path, 
        epochs=50, 
        imgsz=224, 
        batch=8, # Valor menor para não travar o PC
        project="treino_industrial", 
        name="versao_1"
    )

if __name__ == '__main__':
    iniciar_treino()