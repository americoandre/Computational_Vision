import os
import shutil
import time
import uvicorn
import glob
import asyncio
import json # ADICIONADO
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import base64
import cv2
import numpy as np
from typing import List
from datetime import datetime

# --- CONFIGURAÇÃO DE AMBIENTE ---
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

app = FastAPI()

# --- ESTRUTURA DE DIRETÓRIOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CAPTURES_DIR = os.path.join(STATIC_DIR, "capturas")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
TREINO_OUTPUT_DIR = os.path.join(BASE_DIR, "treino_final")

# Categorias do Projeto
CATEGORIAS = ["biologico", "metal", "plastico", "vidro", "papel"]
MAPA_CORES = {
    "biologico": "bg-green-600 hover:bg-green-700 border-green-500",
    "metal": "bg-gray-500 hover:bg-gray-600 border-gray-400",
    "plastico": "bg-blue-600 hover:bg-blue-700 border-blue-500",
    "vidro": "bg-cyan-600 hover:bg-cyan-700 border-cyan-500",
    "papel": "bg-amber-600 hover:bg-amber-700 border-amber-500"
}

# Criação de Infraestrutura
for p in [CAPTURES_DIR, DATASET_DIR, WEIGHTS_DIR, TREINO_OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

# Cria estrutura de pastas do Dataset (train/val)
for split in ["train", "val"]:
    for cat in CATEGORIAS:
        os.makedirs(os.path.join(DATASET_DIR, split, cat), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- GERENCIADOR DE CONEXÕES (WEBSOCKET) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"ESP32-CAM conectado. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"ESP32-CAM desconectado. Restantes: {len(self.active_connections)}")

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    # ADICIONADO: Função broadcast necessária para o novo fluxo
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# --- MANAGERS SEPARADOS ---
cam_manager = ConnectionManager()  # Renomeado de 'manager' para 'cam_manager' para clareza
dev_manager = ConnectionManager()  # ADICIONADO: Manager para ESP32 Dev

# ADICIONADO: Variável global para armazenar o último resultado real da IA
latest_ai_result = {
    "class": "AGUARDANDO",
    "confidence": 0.0,
    "timestamp": 0
}

# --- INTELIGÊNCIA ARTIFICIAL ---
def carregar_inteligencia():
    # Tenta carregar o modelo customizado treinado
    modelo_custom = os.path.join(TREINO_OUTPUT_DIR, "estacao_trabalho", "weights", "best.pt")
    if os.path.exists(modelo_custom):
        print(f"Modelo customizado carregado: {modelo_custom}")
        return YOLO(modelo_custom)
    else:
        print("Modelo padrão YOLOv8n-cls carregado (Ainda não treinado).")
        return YOLO("yolov8n-cls.pt")

brain = carregar_inteligencia()

# Mapa de tradução e correção de classes do modelo base (caso use o modelo pré-treinado genérico)
MAPA_ESTRITO = {
    "thimble": "metal", "can": "metal", "tin": "metal", "hammer": "metal", "wrench": "metal", "nails": "metal",
    "bottle": "vidro", "glass": "vidro", "beaker": "vidro", "wine glass": "vidro",
    "ashcan": "plastico", "water bottle": "plastico", "cup": "plastico", "plastic bag": "plastico",
    "apple": "biologico", "orange": "biologico", "banana": "biologico", "food": "biologico", "fruit": "biologico",
    "box": "papel", "envelope": "papel", "paper": "papel", "carton": "papel", "book": "papel"
}

# --- FUNÇÕES AUXILIARES ---

def get_latest_esp32_image():
    """Retorna a imagem mais recente do ESP32-CAM na pasta temporária."""
    try:
        esp32_images = glob.glob(os.path.join(CAPTURES_DIR, "esp32cam_*.jpg"))
        if esp32_images:
            latest = max(esp32_images, key=os.path.getctime)
            return os.path.basename(latest)
    except Exception as e:
        print(f"Erro ao buscar imagem ESP32: {e}")
    return None

def get_dataset_history():
    """Varre a pasta dataset/train para mostrar o histórico de ensinamentos."""
    history = []
    try:
        # Busca em todas as categorias dentro de train
        for categoria in CATEGORIAS:
            cat_path = os.path.join(DATASET_DIR, "train", categoria)
            if os.path.exists(cat_path):
                files = glob.glob(os.path.join(cat_path, "*.jpg")) + \
                        glob.glob(os.path.join(cat_path, "*.png"))
                
                for f in sorted(files, key=os.path.getctime, reverse=True):
                    filename = os.path.basename(f)
                    # URL relativa para acessar a imagem
                    static_view_path = os.path.join(STATIC_DIR, "dataset_images", categoria)
                    os.makedirs(static_view_path, exist_ok=True)
                    static_file = os.path.join(static_view_path, filename)
                    
                    # Garante que a imagem exista em static para ser servida
                    if not os.path.exists(static_file):
                        shutil.copy(f, static_file)
                    
                    history.append({
                        "filename": filename,
                        "class": categoria,
                        "url": f"/static/dataset_images/{categoria}/{filename}",
                        "date": datetime.fromtimestamp(os.path.getctime(f)).strftime("%d/%m/%Y %H:%M")
                    })
    except Exception as e:
        print(f"Erro ao montar histórico: {e}")
    
    # Ordenar por data (mais recente primeiro)
    return sorted(history, key=lambda x: x['date'], reverse=True)

# --- ROTAS DA API ---

@app.get("/api/esp32-latest")
async def get_latest_image_api():
    global latest_ai_result # ADICIONADO
    latest = get_latest_esp32_image()
    if latest:
        return JSONResponse(content={
            "filename": latest,
            "url": f"/static/capturas/{latest}",
            "result": latest_ai_result # ADICIONADO: Envia o resultado real
        })
    return JSONResponse(content={"error": "Sem imagem"}, status_code=404)

@app.get("/api/dataset-history")
async def get_history_api():
    """Retorna o histórico de imagens ensinadas."""
    data = get_dataset_history()
    return JSONResponse(content={"images": data})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recebe uma imagem, salva temporariamente, prediz e retorna."""
    # Gera nome único
    ext = os.path.splitext(file.filename)[1]
    fname = f"temp_{int(time.time())}{ext}"
    path = os.path.join(CAPTURES_DIR, fname)
    
    # Salva arquivo
    with open(path, "wb") as b:
        shutil.copyfileobj(file.file, b)
    
    # Prediz
    results = brain.predict(source=path, imgsz=224, conf=0.01, verbose=False)
    
    try:
        cid = int(results[0].probs.top1)
        n_en = results[0].names[cid].lower()
        
        # Verifica se já é uma das nossas categorias
        if n_en in CATEGORIAS:
            res = n_en
        else:
            # Tenta mapear com o conhecimento estrito
            res = MAPA_ESTRITO.get(n_en, "plastico")
    except Exception as e:
        print(f"Erro na predição: {e}")
        res = "plastico" # Fallback
        
    return {"resultado": res.upper(), "filename": fname, "image_url": f"/static/capturas/{fname}"}

@app.post("/teach")
async def teach(filename: str = Form(...), correct_class: str = Form(...)):
    """
    Move a imagem predita para a pasta de treino da classe correta.
    Isso ensina a IA para o próximo ciclo de treino.
    """
    src = os.path.join(CAPTURES_DIR, filename)
    
    # Caminhos de destino
    dest_train = os.path.join(DATASET_DIR, "train", correct_class, filename)
    
    if os.path.exists(src):
        # Move para treino
        shutil.move(src, dest_train)
        
        # Copia para visualização no histórico imediatamente
        hist_dir = os.path.join(STATIC_DIR, "dataset_images", correct_class)
        os.makedirs(hist_dir, exist_ok=True)
        shutil.copy(dest_train, os.path.join(hist_dir, filename))
        
        return {"status": "success", "message": f"Imagem ensinada como {correct_class.upper()}"}
    
    return {"status": "error", "message": "Arquivo não encontrado"}

@app.post("/run_train")
async def run_train():
    """
    Executa o fine-tuning do modelo com as imagens ensinadas.
    """
    global brain
    
    # Verifica se há dados suficientes
    count = sum(len(files) for r, d, files in os.walk(os.path.join(DATASET_DIR, "train")))
    if count < 5:
        return {"status": "error", "message": "É preciso ensinar pelo menos 5 imagens antes de treinar."}
    
    try:
        # Usamos ThreadPoolExecutor para não bloquear o event loop do FastAPI completamente
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, lambda: brain.train(
                data=DATASET_DIR, 
                epochs=30, 
                imgsz=224, 
                batch=8, 
                project=TREINO_OUTPUT_DIR, 
                name="estacao_trabalho",
                verbose=True
            ))
        
        # Recarrega o modelo
        brain = carregar_inteligencia()
        return {"status": "success", "message": "Treino concluído e modelo atualizado!"}
    except Exception as e:
        print(f"Erro no treino: {e}")
        return {"status": "error", "message": str(e)}

# --- NOVO WEBSOCKET (ESP32 DEV SENSOR) ---
@app.websocket("/ws/esp32dev")
async def websocket_dev_endpoint(websocket: WebSocket):
    await dev_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if not data: continue
            
            try:
                # Decodifica mensagem JSON do ESP32 Dev
                msg = json.loads(data)
                sensor_active = msg.get("sensor_active")
                
                # Lógica de Controle da Câmera
                if sensor_active == True:
                    print("Sensor Ativo -> Avisando Câmera para MODO RÁPIDO")
                    await cam_manager.broadcast('{"mode": "fast"}')
                elif sensor_active == False:
                    print("Sensor Inativo -> Avisando Câmera para MODO LENTO")
                    await cam_manager.broadcast('{"mode": "slow"}')
                    
            except json.JSONDecodeError:
                print("Erro JSON do ESP32 Dev")
                
    except WebSocketDisconnect:
        dev_manager.disconnect(websocket)
    except Exception as e:
        print(f"Erro geral WebSocket Dev: {e}")
        dev_manager.disconnect(websocket)

# --- WEBSOCKET (ESP32-CAM) ---
@app.websocket("/ws/esp32cam")
async def websocket_endpoint(websocket: WebSocket):
    await cam_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if not data: continue
            
            try:
                # Decodifica imagem base64
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None: continue
                
                # Salva captura
                timestamp = int(time.time())
                filename = f"esp32cam_{timestamp}.jpg"
                path = os.path.join(CAPTURES_DIR, filename)
                cv2.imwrite(path, img)
                
                # Predição
                results = brain.predict(source=path, imgsz=224, conf=0.01, verbose=False)
                
                final_class = "plastico" # Fallback
                confidence = 0.0
                
                # --- ADICIONADO: Lógica detalhada de predição ---
                try:
                    cid = int(results[0].probs.top1)
                    raw_name = results[0].names[cid].lower()
                    confidence = float(results[0].probs.top1conf)
                    
                    if raw_name in CATEGORIAS:
                        final_class = raw_name
                    else:
                        mapped = MAPA_ESTRITO.get(raw_name)
                        if mapped:
                            final_class = mapped
                        else:
                            final_class = "plastico"
                        
                    print(f"   [IA] Classificado como: {final_class.upper()}")
                    
                except Exception as e:
                    print(f"   [IA] Erro no processo de predição: {e}")

                # ADICIONADO: Atualiza variável global
                global latest_ai_result
                latest_ai_result = {
                    "class": final_class.upper(),
                    "confidence": round(confidence, 2),
                    "timestamp": timestamp
                }
                
                # ADICIONADO: Envia resultado para ESP32-Dev
                response_dev = {
                    "target": "dev",
                    "action": "process_result",
                    "result_class": final_class.upper(),
                    "confidence": confidence
                }
                if dev_manager.active_connections:
                    await dev_manager.broadcast(json.dumps(response_dev))
                
                # Resposta para o ESP32-CAM
                response = {
                    "status": "processed",
                    "result": final_class.upper(),
                    "timestamp": timestamp
                }
                await cam_manager.send_message(str(response), websocket)
                
            except Exception as e:
                print(f"Erro no processamento WebSocket: {e}")
                
    except WebSocketDisconnect:
        cam_manager.disconnect(websocket)
    except Exception as e:
        print(f"Erro geral WebSocket: {e}")
        cam_manager.disconnect(websocket)

# --- INTERFACE WEB (DASHBOARD) ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    latest_image = get_latest_esp32_image()
    latest_url = f"/static/capturas/{latest_image}" if latest_image else "/static/placeholder.jpg"
    
    # Geração dos botões de classe para o Treinamento
    botoes_classe_html = ""
    for cat in CATEGORIAS:
        cor_classe = MAPA_CORES.get(cat, "bg-gray-600")
        botoes_classe_html += f"""
        <button onclick="corrigirClasse('{cat}')" 
                class="w-full py-4 px-6 rounded-xl text-white font-bold text-lg shadow-lg transform transition hover:scale-105 active:scale-95 border-b-4 {cor_classe}">
            {cat.upper()}
        </button>
        """

    content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard Industrial</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #e2e8f0; }}
            .glass {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }}
            .nav-btn.active {{ background: rgba(56, 189, 248, 0.1); color: #38bdf8; border-color: #0ea5e9; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; animation: fadeIn 0.3s ease; }}
            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            
            /* Estilização do Input File escondido mas bonito */
            .file-upload-wrapper {{ position: relative; overflow: hidden; display: inline-block; }}
            .file-upload-wrapper input[type=file] {{ font-size: 100px; position: absolute; left: 0; top: 0; opacity: 0; cursor: pointer; }}
        </style>
    </head>
    <body class="min-h-screen flex flex-col">
        
        <!-- HEADER -->
        <header class="glass sticky top-0 z-50 border-b border-slate-700">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex items-center justify-between h-16">
                    <div class="flex items-center gap-3">
                        <!-- Ícone Removido conforme solicitado -->
                        <span class="font-bold text-xl tracking-tight">Central de Processamento de Dados e Aprendizado</span>
                    </div>
                    <div class="flex items-center gap-4">
                        <div class="hidden md:flex items-center gap-2 text-sm text-slate-400">
                            <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            Sistema Online
                        </div>
                        <div id="clock" class="font-mono text-slate-400 text-sm">00:00:00</div>
                    </div>
                </div>
            </div>
        </header>

        <!-- NAVEGAÇÃO -->
        <nav class="max-w-7xl mx-auto px-4 mt-6">
            <div class="flex gap-2 overflow-x-auto pb-2">
                <button onclick="switchTab('monitoramento')" id="btn-monitoramento" class="nav-btn active px-5 py-2 rounded-lg border border-transparent font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition">
                    <i class="fas fa-video mr-2"></i>Monitoramento
                </button>
                <button onclick="switchTab('treinamento')" id="btn-treinamento" class="nav-btn px-5 py-2 rounded-lg border border-transparent font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition">
                    <i class="fas fa-graduation-cap mr-2"></i>Treinamento & Correção
                </button>
                <button onclick="switchTab('historico')" id="btn-historico" class="nav-btn px-5 py-2 rounded-lg border border-transparent font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition">
                    <i class="fas fa-history mr-2"></i>Histórico Ensinado
                </button>
                <button onclick="switchTab('config')" id="btn-config" class="nav-btn px-5 py-2 rounded-lg border border-transparent font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition">
                    <i class="fas fa-cogs mr-2"></i>Configurações
                </button>
            </div>
        </nav>

        <!-- CONTEÚDO PRINCIPAL -->
        <main class="flex-1 max-w-7xl mx-auto px-4 py-6 w-full">
            
            <!-- TAB: MONITORAMENTO -->
            <div id="tab-monitoramento" class="tab-content active">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Coluna Principal: Câmera -->
                    <div class="lg:col-span-2 glass rounded-2xl p-1">
                        <div class="relative bg-black rounded-xl overflow-hidden aspect-video">
                            <img id="esp32Image" src="{latest_url}" class="w-full h-full object-contain" alt="Câmera">
                            <div class="absolute top-4 left-4 bg-black/60 px-3 py-1 rounded text-xs font-mono text-white backdrop-blur-sm">
                                <span class="text-green-400">●</span> LIVE FEED
                            </div>
                            <div class="absolute bottom-4 left-4 right-4 flex justify-between items-end">
                                <div>
                                    <div class="text-xs text-slate-400">Classificação Atual</div>
                                    <div id="liveClass" class="text-2xl font-bold text-white">--</div>
                                </div>
                                <button onclick="refreshMonitor()" class="bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-full transition">
                                    <i class="fas fa-sync-alt"></i>
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Coluna Lateral: Stats -->
                    <div class="space-y-6">
                        <div class="glass rounded-2xl p-6">
                            <h3 class="text-lg font-semibold text-white mb-4">Status Operacional</h3>
                            <div class="space-y-4">
                                <div class="flex justify-between items-center">
                                    <span class="text-slate-400">Conexão ESP32</span>
                                    <span class="text-green-400 font-medium">Estável</span>
                                </div>
                                <div class="flex justify-between items-center">
                                    <span class="text-slate-400">Latência</span>
                                    <span class="text-white font-medium">< 50ms</span>
                                </div>
                                <div class="flex justify-between items-center">
                                    <span class="text-slate-400">Imagens Processadas</span>
                                    <span id="totalProcessed" class="text-white font-medium">0</span>
                                </div>
                            </div>
                        </div>

                        <div class="glass rounded-2xl p-6">
                            <h3 class="text-lg font-semibold text-white mb-2">Ações Rápidas</h3>
                            <p class="text-sm text-slate-400 mb-4">Gerencie o fluxo da linha de produção.</p>
                            <div class="grid grid-cols-2 gap-3">
                                <button onclick="switchTab('treinamento')" class="bg-slate-700 hover:bg-slate-600 p-3 rounded-xl text-white text-sm font-medium transition">
                                    <i class="fas fa-camera mb-1 block text-lg"></i>
                                    Capturar p/ Treino
                                </button>
                                <button onclick="alert('Função de pausa implementada no ESP32')" class="bg-slate-700 hover:bg-slate-600 p-3 rounded-xl text-white text-sm font-medium transition">
                                    <i class="fas fa-pause mb-1 block text-lg"></i>
                                    Pausar Linha
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB: TREINAMENTO -->
            <div id="tab-treinamento" class="tab-content">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    
                    <!-- Área de Upload e Visualização -->
                    <div class="space-y-6">
                        <div class="glass rounded-2xl p-6 text-center">
                            <h2 class="text-xl font-bold text-white mb-6">Nova Entrada de Dados</h2>
                            
                            <!-- Área de Drag & Drop / Clique -->
                            <div class="file-upload-wrapper w-full group cursor-pointer border-2 border-dashed border-slate-600 hover:border-blue-500 rounded-xl p-8 transition-colors bg-slate-800/50">
                                <input type="file" id="trainInput" accept="image/*" capture="environment" onchange="handleTrainUpload(this)">
                                <div class="text-slate-400 group-hover:text-blue-400 transition-colors">
                                    <i class="fas fa-cloud-upload-alt text-4xl mb-3"></i>
                                    <p class="text-lg font-medium">Toque para Capturar ou Enviar Foto</p>
                                    <p class="text-sm opacity-60 mt-1">Suporta JPG, PNG (Mobile ou Desktop)</p>
                                </div>
                            </div>

                            <!-- Pré-visualização e Predição -->
                            <div id="trainPreviewArea" class="mt-6 hidden">
                                <div class="flex gap-4 items-start bg-slate-800/50 p-4 rounded-xl border border-slate-700">
                                    <img id="trainPreviewImg" src="" class="w-32 h-32 object-cover rounded-lg border border-slate-600 bg-black">
                                    <div class="flex-1">
                                        <div class="flex justify-between items-center mb-2">
                                            <span class="text-xs font-mono text-slate-400" id="trainFilename">imagem.jpg</span>
                                            <span class="text-xs bg-blue-900 text-blue-200 px-2 py-0.5 rounded">IA PREDIÇÃO</span>
                                        </div>
                                        <div class="text-slate-400 text-sm mb-1">O sistema identificou como:</div>
                                        <div id="trainPrediction" class="text-2xl font-bold text-white">---</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Área de Correção -->
                    <div class="space-y-6">
                        <div class="glass rounded-2xl p-6 h-full flex flex-col">
                            <h2 class="text-xl font-bold text-white mb-2">Correção e Ensino</h2>
                            <p class="text-slate-400 text-sm mb-6">Se a predição acima estiver incorreta, selecione a classe verdadeira para ensinar a IA.</p>
                            
                            <div id="correctionButtons" class="grid grid-cols-1 gap-4 flex-1 opacity-50 pointer-events-none transition-opacity">
                                {botoes_classe_html}
                            </div>

                            <div class="mt-6 pt-6 border-t border-slate-700">
                                <button onclick="runTraining()" id="trainModelBtn" class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-4 rounded-xl shadow-lg transition transform hover:-translate-y-1">
                                    <i class="fas fa-brain mr-2"></i> ATUALIZAR MODELO COM NOVOS DADOS
                                </button>
                                <div class="text-center mt-3">
                                    <button onclick="resetTrainUI()" class="text-sm text-slate-500 hover:text-slate-300 underline">Limpar e Começar Nova Imagem</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB: HISTÓRICO -->
            <div id="tab-historico" class="tab-content">
                <div class="glass rounded-2xl p-6">
                    <div class="flex flex-col md:flex-row justify-between items-center mb-8 gap-4">
                        <div>
                            <h2 class="text-2xl font-bold text-white">Base de Conhecimento</h2>
                            <p class="text-slate-400">Imagens classificadas e ensinadas à IA.</p>
                        </div>
                        <div class="flex gap-2 overflow-x-auto pb-2 w-full md:w-auto" id="historyFilters">
                            <button onclick="filterHistory('all')" class="filter-btn active px-4 py-2 rounded-lg bg-slate-700 text-white text-sm font-medium whitespace-nowrap">Todas</button>
                            <!-- Botões gerados via JS -->
                        </div>
                    </div>

                    <div id="historyGrid" class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                        <!-- Cards injetados via JS -->
                    </div>
                    <div id="emptyHistory" class="text-center py-12 text-slate-500 hidden">
                        <i class="fas fa-folder-open text-4xl mb-3"></i>
                        <p>Nenhuma imagem ensinada nesta categoria ainda.</p>
                    </div>
                </div>
            </div>

            <!-- TAB: CONFIGURAÇÕES -->
            <div id="tab-config" class="tab-content">
                <div class="glass rounded-2xl p-8">
                    <h2 class="text-2xl font-bold text-white mb-6">Configurações do Sistema</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div>
                            <h3 class="text-lg font-semibold text-blue-400 mb-4">Parâmetros de Treino</h3>
                            <div class="space-y-4">
                                <div>
                                    <label class="text-slate-400 text-sm">Épocas (Ciclos de Aprendizado)</label>
                                    <input type="number" value="30" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white">
                                </div>
                                <div>
                                    <label class="text-slate-400 text-sm">Tamanho da Imagem (Imgsz)</label>
                                    <select class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white">
                                        <option value="224">224x224 (Rápido)</option>
                                        <option value="640">640x640 (Preciso)</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <div>
                            <h3 class="text-lg font-semibold text-red-400 mb-4">Zona de Perigo</h3>
                            <div class="bg-red-900/20 border border-red-900/50 rounded-xl p-6">
                                <p class="text-slate-300 mb-4">Resetar o sistema apagará todo o histórico de aprendizado e voltará ao modelo padrão de fábrica.</p>
                                <button onclick="if(confirm('Tem certeza?')){{ location.reload() }}" class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium">
                                    Resetar Sistema Completo
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </main>

        <footer class="border-t border-slate-800 mt-8 py-6">
            <div class="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
                &copy; 2023 Sistema de Classificação Industrial v2.0 - Processamento Edge AI
            </div>
        </footer>

        <!-- SCRIPTS -->
        <script>
            // --- ESTADO GLOBAL ---
            let currentTrainFile = null;
            let historyData = [];

            // --- NAVEGAÇÃO ---
            function switchTab(tabId) {{
                // Esconde todas
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
                
                // Mostra alvo
                document.getElementById('tab-' + tabId).classList.add('active');
                document.getElementById('btn-' + tabId).classList.add('active');

                // Ações específicas
                if (tabId === 'monitoramento') refreshMonitor();
                if (tabId === 'historico') loadHistory();
            }}

            // --- RELÓGIO ---
            setInterval(() => {{
                const now = new Date();
                document.getElementById('clock').textContent = now.toLocaleTimeString('pt-BR');
            }}, 1000);

            // --- MONITORAMENTO ---
            let lastImageSrc = "";
            
            async function refreshMonitor() {{
                try {{
                    const res = await fetch('/api/esp32-latest');
                    const data = await res.json();
                    
                    if (!data.error) {{
                        const img = document.getElementById('esp32Image');
                        // Atualiza apenas se o arquivo mudou para evitar flicker
                        if (img.src.indexOf(data.filename) === -1) {{
                            img.src = data.url;
                            
                            // Simula atualização da classe (lógica simplificada para UI)
                            document.getElementById('liveClass').innerText = "ANALISANDO...";
                            setTimeout(() => {{
                                // CORREÇÃO AQUI: Usa a variável 'data' correta do fetch e o {{}} correto
                                const classEl = document.getElementById('liveClass');
                                if(data.result && data.result.class) {{
                                    classEl.innerText = data.result.class;
                                    // Muda cor baseada na classe
                                    let color = "text-white";
                                    if(data.result.class === "VIDRO") color = "text-cyan-400";
                                    if(data.result.class === "PLASTICO") color = "text-blue-400";
                                    if(data.result.class === "METAL") color = "text-gray-400";
                                    if(data.result.class === "BIOLOGICO") color = "text-green-400";
                                    if(data.result.class === "PAPEL") color = "text-amber-400";
                                    classEl.className = `text-2xl font-bold ${{color}}`;
                                }} else {{
                                    classEl.innerText = "--";
                                }}
                            }}, 500);
                        }}
                    }}
                }} catch (e) {{
                    console.error("Erro no monitor:", e);
                }}
            }}
            
            // Atualiza monitor a cada 2 segundos
            setInterval(refreshMonitor, 2000);

            // --- LÓGICA DE TREINAMENTO ---
            async function handleTrainUpload(input) {{
                if (input.files && input.files[0]) {{
                    const file = input.files[0];
                    const formData = new FormData();
                    formData.append('file', file);

                    // UI Loading
                    const btnArea = document.getElementById('correctionButtons');
                    btnArea.innerHTML = '<div class="col-span-1 text-center py-10"><i class="fas fa-circle-notch fa-spin text-3xl text-blue-500"></i><p class="text-slate-400 mt-2">Analisando imagem...</p></div>';

                    try {{
                        // Envia para API prever
                        const res = await fetch('/predict', {{ method: 'POST', body: formData }});
                        const data = await res.json();

                        // Atualiza UI com resultado
                        currentTrainFile = data.filename;
                        document.getElementById('trainFilename').innerText = data.filename;
                        document.getElementById('trainPrediction').innerText = data.resultado;
                        document.getElementById('trainPreviewImg').src = data.image_url;
                        document.getElementById('trainPreviewArea').classList.remove('hidden');

                        // Gera botões de correção
                        const classes = ["biologico", "metal", "plastico", "vidro", "papel"];
                        
                        let buttonsHtml = "";
                        classes.forEach(cat => {{
                            const isMatch = cat.toUpperCase() === data.resultado;
                            const colorClass = isMatch ? 'ring-2 ring-white scale-105' : 'opacity-80 hover:opacity-100';
                            
                            // Mapeamento de cores para botões
                            let btnColor = "bg-slate-600";
                            if(cat === 'biologico') btnColor = "bg-green-600 hover:bg-green-700";
                            if(cat === 'metal') btnColor = "bg-gray-500 hover:bg-gray-600";
                            if(cat === 'plastico') btnColor = "bg-blue-600 hover:bg-blue-700";
                            if(cat === 'vidro') btnColor = "bg-cyan-600 hover:bg-cyan-700";
                            if(cat === 'papel') btnColor = "bg-amber-600 hover:bg-amber-700";

                            buttonsHtml += `
                            <button onclick="corrigirClasse('${cat}')" 
                                    class="w-full py-4 px-6 rounded-xl text-white font-bold text-lg shadow-lg transform transition hover:scale-105 active:scale-95 border-b-4 border-black/20 {{btnColor}} {{colorClass}}">
                                <div class="flex justify-between items-center">
                                    <span>${{cat.toUpperCase()}}</span>
                                    ${{isMatch ? '<i class="fas fa-check-circle"></i>' : ''}}
                                </div>
                            </button>`;
                        }});

                        btnArea.innerHTML = buttonsHtml;
                        btnArea.classList.remove('opacity-50', 'pointer-events-none');

                    }} catch (e) {{
                        alert('Erro ao processar imagem.');
                        console.error(e);
                        resetTrainUI();
                    }}
                }}
            }}

            async function corrigirClasse(classe) {{
                if (!currentTrainFile) return;

                const params = new URLSearchParams();
                params.append('filename', currentTrainFile);
                params.append('correct_class', classe);

                try {{
                    const res = await fetch('/teach', {{ method: 'POST', body: params }});
                    const data = await res.json();

                    if (data.status === 'success') {{
                        // Feedback visual de sucesso
                        const btnArea = document.getElementById('correctionButtons');
                        btnArea.innerHTML = `
                            <div class="col-span-1 text-center py-10 bg-green-900/30 rounded-xl border border-green-500/50">
                                <i class="fas fa-check-circle text-5xl text-green-500 mb-3"></i>
                                <h3 class="text-xl font-bold text-white">Salvo!</h3>
                                <p class="text-green-200">Ensinado como ${{classe.toUpperCase()}}</p>
                            </div>
                        `;
                        
                        // Atualiza contador de imagens processadas (simbólico)
                        const counter = document.getElementById('totalProcessed');
                        counter.innerText = parseInt(counter.innerText) + 1;
                    }}
                }} catch (e) {{
                    alert("Erro ao salvar ensinamento.");
                }}
            }}

            function resetTrainUI() {{
                document.getElementById('trainInput').value = "";
                document.getElementById('trainPreviewArea').classList.add('hidden');
                document.getElementById('correctionButtons').classList.add('opacity-50', 'pointer-events-none');
                document.getElementById('correctionButtons').innerHTML = '<div class="text-center text-slate-500 py-10">Aguardando imagem...</div>';
                currentTrainFile = null;
            }}

            async function runTraining() {{
                const btn = document.getElementById('trainModelBtn');
                const originalText = btn.innerHTML;
                
                btn.disabled = true;
                btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> TREINANDO... (ISSO PODE LEVAR MINUTOS)';

                try {{
                    const res = await fetch('/run_train', {{ method: 'POST' }});
                    const data = await res.json();
                    
                    if (data.status === 'success') {{
                        alert("Modelo atualizado com sucesso! A IA agora utiliza os novos conhecimentos.");
                    }} else {{
                        alert("Erro: " + data.message);
                    }}
                }} catch (e) {{
                    alert("Erro de comunicação no treino.");
                }} finally {{
                    btn.disabled = false;
                    btn.innerHTML = originalText;
                }}
            }}

            // --- HISTÓRICO ---
            let currentFilter = 'all';

            async function loadHistory() {{
                // Gera filtros
                const filterContainer = document.getElementById('historyFilters');
                const classes = ["all", "biologico", "metal", "plastico", "vidro", "papel"];
                
                let filterHtml = `<button onclick="filterHistory('all')" class="filter-btn px-4 py-2 rounded-lg bg-slate-700 text-white text-sm font-medium whitespace-nowrap transition hover:bg-slate-600">Todas</button>`;
                classes.slice(1).forEach(c => {{
                    filterHtml += `<button onclick="filterHistory('${{c}}')" class="filter-btn px-4 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 text-sm font-medium whitespace-nowrap transition hover:border-slate-500 capitalize">${{c}}</button>`;
                }});
                filterContainer.innerHTML = filterHtml;

                // Busca dados
                const res = await fetch('/api/dataset-history');
                const json = await res.json();
                historyData = json.images;
                
                renderHistory();
            }}

            function filterHistory(category) {{
                currentFilter = category;
                
                // Atualiza visual dos botões de filtro
                const buttons = document.querySelectorAll('.filter-btn');
                buttons.forEach(b => {{
                    if (b.innerText.toLowerCase().includes(category) || (category === 'all' && b.innerText === 'Todas')) {{
                        b.classList.remove('bg-slate-800', 'text-slate-300', 'border-slate-700');
                        b.classList.add('bg-blue-600', 'text-white', 'border-blue-500');
                    }} else {{
                        b.classList.add('bg-slate-800', 'text-slate-300', 'border-slate-700');
                        b.classList.remove('bg-blue-600', 'text-white', 'border-blue-500');
                    }}
                }});

                renderHistory();
            }}

            function renderHistory() {{
                const grid = document.getElementById('historyGrid');
                const empty = document.getElementById('emptyHistory');
                
                const filtered = currentFilter === 'all' 
                    ? historyData 
                    : historyData.filter(img => img.class === currentFilter);

                if (filtered.length === 0) {{
                    grid.innerHTML = '';
                    empty.classList.remove('hidden');
                    return;
                }}

                empty.classList.add('hidden');
                
                grid.innerHTML = filtered.map(img => {{
                    let badgeColor = "bg-slate-600";
                    if(img.class === 'biologico') badgeColor = "bg-green-600";
                    if(img.class === 'metal') badgeColor = "bg-gray-500";
                    if(img.class === 'plastico') badgeColor = "bg-blue-600";
                    if(img.class === 'vidro') badgeColor = "bg-cyan-600";
                    if(img.class === 'papel') badgeColor = "bg-amber-600";

                    return `
                    <div class="bg-slate-800 rounded-lg overflow-hidden border border-slate-700 group hover:border-slate-500 transition">
                        <div class="relative aspect-square bg-black">
                            <img src="${{img.url}}" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition" loading="lazy">
                            <div class="absolute top-2 right-2 {{badgeColor}} text-white text-[10px] font-bold px-2 py-0.5 rounded uppercase shadow">
                                ${{img.class}}
                            </div>
                        </div>
                        <div class="p-2">
                            <div class="text-[10px] text-slate-400 font-mono truncate">${{img.date}}</div>
                        </div>
                    </div>
                    `;
                }}).join('');
            }}

            // Inicia na aba de monitoramento
            document.addEventListener('DOMContentLoaded', () => {{
                switchTab('monitoramento');
            }});

        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
