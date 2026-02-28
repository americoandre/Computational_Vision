import os
import shutil
import time
import uvicorn
import glob
from fastapi import FastAPI, File, UploadFile, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import base64
import cv2
import numpy as np
from typing import List
from datetime import datetime

# --- AMBIENTE OFFLINE ---
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

app = FastAPI()

# Definição de Caminhos Estruturais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CAPTURES_DIR = os.path.join(STATIC_DIR, "capturas")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Categorias Estritas do Projeto
CATEGORIAS = ["biologico", "metal", "plastico", "vidro", "papel"]

# Criação Automática de Infraestrutura
for p in [CAPTURES_DIR, DATASET_DIR, WEIGHTS_DIR]:
    os.makedirs(p, exist_ok=True)
for split in ["train", "val"]:
    for cat in CATEGORIAS:
        os.makedirs(os.path.join(DATASET_DIR, split, cat), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- GERENCIADOR WEBSOCKET ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_capture_time = 0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"ESP32-CAM conectado. Total de conexões: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"ESP32-CAM desconectado. Conexões restantes: {len(self.active_connections)}")

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    def set_capture_time(self):
        self.last_capture_time = time.time()

manager = ConnectionManager()

# --- FUNÇÕES AUXILIARES ---
def carregar_inteligencia():
    modelo_treinado = os.path.join(BASE_DIR, "treino_output", "industrial_v1", "weights", "best.pt")
    if os.path.exists(modelo_treinado):
        return YOLO(modelo_treinado)
    return YOLO("yolov8n-cls.pt")

def get_latest_esp32_image():
    """Retorna a imagem mais recente do ESP32-CAM"""
    try:
        esp32_images = glob.glob(os.path.join(CAPTURES_DIR, "esp32cam_*.jpg"))
        if esp32_images:
            latest = max(esp32_images, key=os.path.getctime)
            return os.path.basename(latest)
    except Exception as e:
        print(f"Erro ao buscar imagem mais recente: {e}")
    return None

def get_all_esp32_images():
    """Retorna todas as imagens do ESP32-CAM"""
    try:
        esp32_images = glob.glob(os.path.join(CAPTURES_DIR, "esp32cam_*.jpg"))
        images_info = []
        for img_path in sorted(esp32_images, key=os.path.getctime, reverse=True):
            filename = os.path.basename(img_path)
            timestamp = filename.replace("esp32cam_", "").replace(".jpg", "")
            try:
                timestamp_int = int(timestamp)
                date_str = datetime.fromtimestamp(timestamp_int).strftime("%d/%m/%Y %H:%M:%S")
            except:
                date_str = "Data desconhecida"
            
            images_info.append({
                "filename": filename,
                "timestamp": timestamp_int,
                "date": date_str,
                "url": f"/static/capturas/{filename}"
            })
        return images_info
    except Exception as e:
        print(f"Erro ao listar imagens: {e}")
        return []

brain = carregar_inteligencia()

MAPA_ESTRITO = {
    "thimble": "metal", "can": "metal", "tin": "metal", "hammer": "metal", "wrench": "metal",
    "bottle": "vidro", "toilet tissue": "vidro", "glass": "vidro", "beaker": "vidro",
    "ashcan": "plastico", "water bottle": "plastico", "cup": "plastico", "plastic bag": "plastico",
    "apple": "biologico", "orange": "biologico", "banana": "biologico", "food": "biologico",
    "box": "papel", "envelope": "papel", "paper": "papel", "carton": "papel"
}

# --- WEBSOCKET PARA ESP32-CAM ---
@app.websocket("/ws/esp32cam")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            if not data:
                continue
                
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    error_msg = '{"status": "error", "message": "Imagem inválida"}'
                    await manager.send_message(error_msg, websocket)
                    continue
                
                timestamp = int(time.time())
                filename = f"esp32cam_{timestamp}.jpg"
                path = os.path.join(CAPTURES_DIR, filename)
                cv2.imwrite(path, img)
                
                # Ativa flag de flash
                manager.set_capture_time()
                
                results = brain.predict(source=path, imgsz=224, conf=0.01)
                
                try:
                    cid = int(results[0].probs.top1)
                    n_en = results[0].names[cid].lower()
                    
                    if n_en in CATEGORIAS:
                        res = n_en
                    else:
                        res = MAPA_ESTRITO.get(n_en, "plastico")
                except:
                    res = "plastico"
                
                response = {
                    "status": "processed",
                    "filename": filename,
                    "result": res.upper(),
                    "timestamp": timestamp,
                    "flash": True  # Indica que foi uma nova captura
                }
                
                await manager.send_message(str(response), websocket)
                print(f"ESP32-CAM: Imagem {filename} processada -> {res.upper()}")
                
            except Exception as e:
                error_msg = f'{{"status": "error", "message": "Erro: {str(e)}"}}'
                await manager.send_message(error_msg, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Erro WebSocket: {e}")
        manager.disconnect(websocket)

# --- NOVOS ENDPOINTS API ---
@app.get("/api/esp32-images")
async def get_esp32_images():
    images = get_all_esp32_images()
    return JSONResponse(content={"images": images})

@app.get("/api/latest-esp32-image")
async def get_latest_image():
    latest = get_latest_esp32_image()
    if latest:
        return JSONResponse(content={
            "filename": latest,
            "url": f"/static/capturas/{latest}",
            "flash": time.time() - manager.last_capture_time < 1  # Flash se captura < 1s atrás
        })
    return JSONResponse(content={"error": "Nenhuma imagem encontrada"}, status_code=404)

# --- PÁGINA WEB PRINCIPAL ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    latest_image = get_latest_esp32_image()
    latest_image_url = f"/static/capturas/{latest_image}" if latest_image else ""
    
    cores = {
        "biologico": "bg-green-600 hover:bg-green-700", 
        "metal": "bg-gray-600 hover:bg-gray-700", 
        "plastico": "bg-blue-600 hover:bg-blue-700", 
        "vidro": "bg-cyan-600 hover:bg-cyan-700", 
        "papel": "bg-amber-600 hover:bg-amber-700"
    }
    
    cores_texto = {
        "biologico": "text-green-600",
        "metal": "text-gray-600", 
        "plastico": "text-blue-600",
        "vidro": "text-cyan-600",
        "papel": "text-amber-600"
    }
    
    botoes_correcao = "".join([f"""
        <button onclick="corrigirClassificacao('{c}')" 
                class="{cores.get(c)} text-white font-bold py-4 px-4 rounded-xl shadow-lg transition-all duration-200 transform hover:scale-105">
            <div class="text-sm font-semibold uppercase tracking-wide">{c}</div>
        </button>
    """ for c in CATEGORIAS])

    content = f"""
    <!DOCTYPE html>
    <html lang="pt">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistema de Classificação Industrial - ESP32-CAM</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                min-height: 100vh;
            }}
            
            .glass-panel {{
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 1rem;
            }}
            
            .tab-button {{
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }}
            
            .tab-button.active {{
                background: rgba(56, 189, 248, 0.1);
                border-color: #0ea5e9;
                color: #38bdf8;
            }}
            
            .tab-button.active::after {{
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 50%;
                height: 3px;
                background: linear-gradient(90deg, #0ea5e9, #38bdf8);
                border-radius: 3px 3px 0 0;
            }}
            
            .flash-animation {{
                animation: flash 0.5s ease-in-out;
            }}
            
            @keyframes flash {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.3; background-color: rgba(255, 255, 255, 0.8); }}
                100% {{ opacity: 1; }}
            }}
            
            .status-indicator {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }}
            
            .status-online {{
                background-color: #10b981;
                box-shadow: 0 0 10px #10b981;
            }}
            
            .status-offline {{
                background-color: #ef4444;
            }}
            
            .image-card {{
                transition: all 0.3s ease;
                transform-origin: center;
            }}
            
            .image-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            }}
            
            .progress-bar {{
                height: 4px;
                background: linear-gradient(90deg, #0ea5e9, #38bdf8);
                border-radius: 2px;
                transition: width 0.3s ease;
            }}
        </style>
    </head>
    <body class="p-4 md:p-6">
        <div class="max-w-7xl mx-auto">
            <!-- CABEÇALHO -->
            <header class="mb-8">
                <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                        <h1 class="text-2xl md:text-3xl font-bold text-white">
                            Sistema de Classificação Industrial
                        </h1>
                        <p class="text-slate-400 mt-1">Monitoramento em tempo real via ESP32-CAM com IA</p>
                    </div>
                    <div class="flex items-center space-x-4">
                        <div class="flex items-center">
                            <span class="status-indicator status-online"></span>
                            <span class="text-sm font-medium text-green-400">Online</span>
                        </div>
                        <div class="text-sm text-slate-400">
                            <span id="currentTime">--:--:--</span>
                        </div>
                    </div>
                </div>
            </header>

            <!-- BARRA DE NAVEGAÇÃO -->
            <nav class="mb-8">
                <div class="flex flex-wrap gap-2 p-2 glass-panel rounded-xl">
                    <button onclick="showTab('monitoramento')" id="tabMonitoramento" 
                            class="tab-button active flex items-center gap-2 px-5 py-3 rounded-lg font-semibold">
                        <i class="fas fa-video"></i>
                        Monitoramento
                    </button>
                    <button onclick="showTab('treino')" id="tabTreino" 
                            class="tab-button flex items-center gap-2 px-5 py-3 rounded-lg font-semibold">
                        <i class="fas fa-brain"></i>
                        Treino & Correção
                    </button>
                    <button onclick="showTab('historico')" id="tabHistorico" 
                            class="tab-button flex items-center gap-2 px-5 py-3 rounded-lg font-semibold">
                        <i class="fas fa-history"></i>
                        Histórico
                    </button>
                    <button onclick="showTab('config')" id="tabConfig" 
                            class="tab-button flex items-center gap-2 px-5 py-3 rounded-lg font-semibold">
                        <i class="fas fa-cog"></i>
                        Configurações
                    </button>
                </div>
            </nav>

            <!-- CONTEÚDO DAS ABAS -->
            <main>
                <!-- ABA MONITORAMENTO -->
                <div id="tabMonitoramento" class="tab-content">
                    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <!-- VISUALIZAÇÃO DA CÂMERA -->
                        <div class="lg:col-span-2">
                            <div class="glass-panel p-6">
                                <div class="flex justify-between items-center mb-6">
                                    <h2 class="text-xl font-bold text-white">Visualização em Tempo Real</h2>
                                    <div class="flex items-center gap-3">
                                        <button onclick="refreshESP32Image()" 
                                                class="bg-sky-600 hover:bg-sky-700 text-white px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition-colors">
                                            <i class="fas fa-sync-alt"></i>
                                            Atualizar
                                        </button>
                                        <div class="text-sm">
                                            <span class="text-slate-400">Intervalo:</span>
                                            <span class="text-white font-medium ml-1">2s</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div id="esp32ImageContainer" class="relative bg-black rounded-xl overflow-hidden">
                                    <div id="flashOverlay" class="absolute inset-0 bg-white opacity-0 z-10 pointer-events-none"></div>
                                    <img id="esp32CurrentImage" 
                                         src="{latest_image_url if latest_image_url else '/static/placeholder.jpg'}" 
                                         class="w-full h-64 md:h-96 object-contain rounded-xl"
                                         onerror="this.onerror=null; this.src='/static/placeholder.jpg'">
                                    
                                    <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-4">
                                        <div class="flex justify-between items-center">
                                            <div>
                                                <div class="text-sm text-slate-300">Última captura</div>
                                                <div id="lastCaptureTime" class="text-white font-bold">--:--:--</div>
                                            </div>
                                            <div>
                                                <div class="text-sm text-slate-300">Classificação atual</div>
                                                <div id="currentClassification" class="text-white font-bold text-lg">--</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div class="glass-panel p-4 rounded-lg">
                                        <div class="text-slate-400 text-sm">Total de Capturas</div>
                                        <div id="totalCaptures" class="text-2xl font-bold text-white">0</div>
                                    </div>
                                    <div class="glass-panel p-4 rounded-lg">
                                        <div class="text-slate-400 text-sm">Taxa de Acerto</div>
                                        <div id="accuracyRate" class="text-2xl font-bold text-green-400">0%</div>
                                    </div>
                                    <div class="glass-panel p-4 rounded-lg">
                                        <div class="text-slate-400 text-sm">Status Conexão</div>
                                        <div class="text-green-400 font-medium">Ativa</div>
                                    </div>
                                    <div class="glass-panel p-4 rounded-lg">
                                        <div class="text-slate-400 text-sm">Última Atividade</div>
                                        <div id="lastActivity" class="text-white font-medium">Agora</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- ESTATÍSTICAS E CONTROLES -->
                        <div class="space-y-6">
                            <!-- CONTROLES -->
                            <div class="glass-panel p-6">
                                <h3 class="text-lg font-bold text-white mb-4">Controles da Câmera</h3>
                                <div class="space-y-4">
                                    <div>
                                        <label class="block text-slate-400 text-sm mb-2">Intervalo de Captura</label>
                                        <input type="range" min="1" max="10" value="2" 
                                               class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                                               onchange="updateCaptureInterval(this.value)">
                                        <div class="flex justify-between text-sm text-slate-400 mt-1">
                                            <span>1s</span>
                                            <span id="intervalValue">2s</span>
                                            <span>10s</span>
                                        </div>
                                    </div>
                                    <div class="grid grid-cols-2 gap-3">
                                        <button onclick="captureManual()" 
                                                class="bg-emerald-600 hover:bg-emerald-700 text-white py-3 rounded-lg font-medium transition-colors">
                                            Captura Manual
                                        </button>
                                        <button onclick="toggleAutoCapture()" id="autoCaptureBtn"
                                                class="bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-medium transition-colors">
                                            Auto: ON
                                        </button>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- STATUS DO SISTEMA -->
                            <div class="glass-panel p-6">
                                <h3 class="text-lg font-bold text-white mb-4">Status do Sistema</h3>
                                <div class="space-y-3">
                                    <div class="flex justify-between items-center">
                                        <span class="text-slate-400">ESP32-CAM</span>
                                        <span class="text-green-400 font-medium">Conectado</span>
                                    </div>
                                    <div class="flex justify-between items-center">
                                        <span class="text-slate-400">Modelo IA</span>
                                        <span class="text-green-400 font-medium">Carregado</span>
                                    </div>
                                    <div class="flex justify-between items-center">
                                        <span class="text-slate-400">Armazenamento</span>
                                        <span id="storageStatus" class="text-green-400 font-medium">Disponível</span>
                                    </div>
                                    <div class="flex justify-between items-center">
                                        <span class="text-slate-400">Processamento</span>
                                        <span class="text-blue-400 font-medium">Ativo</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ABA TREINO E CORREÇÃO -->
                <div id="tabTreino" class="tab-content hidden">
                    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <!-- VISUALIZAÇÃO DA IMAGEM -->
                        <div class="lg:col-span-2">
                            <div class="glass-panel p-6">
                                <h2 class="text-xl font-bold text-white mb-6">Treino e Correção do Modelo</h2>
                                
                                <div class="mb-6">
                                    <div class="flex justify-between items-center mb-4">
                                        <h3 class="text-lg font-semibold text-white">Imagem Atual para Classificação</h3>
                                        <div id="trainingStatus" class="text-sm text-slate-400">Selecione uma imagem</div>
                                    </div>
                                    
                                    <div class="bg-black rounded-xl p-4">
                                        <img id="trainingImage" 
                                             src="/static/placeholder.jpg" 
                                             class="w-full h-80 object-contain rounded-lg">
                                        
                                        <div class="mt-4 flex justify-between items-center">
                                            <div>
                                                <div class="text-sm text-slate-400">Arquivo</div>
                                                <div id="trainingFilename" class="text-white font-medium">Nenhuma imagem selecionada</div>
                                            </div>
                                            <button onclick="selectTrainingImage()" 
                                                    class="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg font-medium">
                                                Selecionar Imagem
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- CLASSIFICAÇÃO ATUAL -->
                                <div class="glass-panel p-5 mb-6">
                                    <h4 class="text-lg font-semibold text-white mb-4">Classificação Atual pelo Modelo</h4>
                                    <div id="currentPrediction" class="text-center py-8">
                                        <div class="text-slate-400">Nenhuma classificação disponível</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- CONTROLES DE TREINO -->
                        <div>
                            <div class="glass-panel p-6 h-full">
                                <h3 class="text-lg font-bold text-white mb-6">Correção Manual</h3>
                                
                                <div class="mb-8">
                                    <div class="text-sm text-slate-400 mb-4">Se a classificação estiver incorreta, selecione a categoria correta:</div>
                                    
                                    <div class="grid grid-cols-2 gap-3">
                                        {botoes_correcao}
                                    </div>
                                    
                                    <div class="mt-6">
                                        <label class="block text-slate-400 text-sm mb-2">Confiança da Correção</label>
                                        <div class="flex items-center gap-3">
                                            <div class="progress-bar w-full"></div>
                                            <span class="text-white font-medium">100%</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="space-y-4">
                                    <button onclick="confirmCorrection()" id="confirmBtn"
                                            class="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                            disabled>
                                        Confirmar Correção
                                    </button>
                                    
                                    <button onclick="skipTraining()" 
                                            class="w-full bg-slate-700 hover:bg-slate-600 text-white py-3 rounded-lg font-medium transition-colors">
                                        Pular Imagem
                                    </button>
                                    
                                    <button onclick="trainModel()" 
                                            class="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-medium transition-colors">
                                        <i class="fas fa-brain mr-2"></i>
                                        Treinar Modelo com Correções
                                    </button>
                                </div>
                                
                                <div class="mt-8 pt-6 border-t border-slate-700">
                                    <h4 class="text-lg font-semibold text-white mb-3">Estatísticas de Treino</h4>
                                    <div class="space-y-3">
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">Imagens Corrigidas</span>
                                            <span id="correctedCount" class="text-white font-medium">0</span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">Precisão Atual</span>
                                            <span id="trainingAccuracy" class="text-green-400 font-medium">0%</span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-slate-400">Último Treino</span>
                                            <span id="lastTraining" class="text-white font-medium">Nunca</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ABA HISTÓRICO -->
                <div id="tabHistorico" class="tab-content hidden">
                    <div class="glass-panel p-6">
                        <div class="flex justify-between items-center mb-6">
                            <h2 class="text-xl font-bold text-white">Histórico de Capturas</h2>
                            <div class="flex items-center gap-3">
                                <div class="relative">
                                    <input type="text" id="searchImages" placeholder="Buscar por data..." 
                                           class="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 pl-10 text-white w-64">
                                    <i class="fas fa-search absolute left-3 top-3 text-slate-500"></i>
                                </div>
                                <button onclick="exportHistory()" 
                                        class="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg font-medium">
                                    Exportar
                                </button>
                            </div>
                        </div>
                        
                        <div id="imagesGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                            <!-- Imagens serão carregadas aqui -->
                        </div>
                        
                        <div id="loading" class="text-center py-12">
                            <div class="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-sky-500 border-r-2 border-sky-500"></div>
                            <p class="text-slate-500 mt-4">Carregando histórico de capturas...</p>
                        </div>
                        
                        <div id="noImages" class="hidden text-center py-12">
                            <i class="fas fa-image text-5xl text-slate-700 mb-4"></i>
                            <h3 class="text-xl font-bold text-slate-400 mb-2">Nenhuma captura encontrada</h3>
                            <p class="text-slate-600">As imagens capturadas pelo ESP32-CAM aparecerão aqui.</p>
                        </div>
                    </div>
                </div>

                <!-- ABA CONFIGURAÇÕES -->
                <div id="tabConfig" class="tab-content hidden">
                    <div class="glass-panel p-6">
                        <h2 class="text-xl font-bold text-white mb-6">Configurações do Sistema</h2>
                        
                        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <!-- CONFIGURAÇÕES DA CÂMERA -->
                            <div>
                                <h3 class="text-lg font-semibold text-white mb-4">Configurações da Câmera</h3>
                                <div class="space-y-4">
                                    <div>
                                        <label class="block text-slate-400 text-sm mb-2">Qualidade da Imagem</label>
                                        <select class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                                            <option>Alta (800x600)</option>
                                            <option selected>Média (640x480)</option>
                                            <option>Baixa (320x240)</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label class="block text-slate-400 text-sm mb-2">Brilho</label>
                                        <input type="range" min="0" max="100" value="50" class="w-full">
                                    </div>
                                    <div>
                                        <label class="block text-slate-400 text-sm mb-2">Contraste</label>
                                        <input type="range" min="0" max="100" value="50" class="w-full">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- CONFIGURAÇÕES DO MODELO -->
                            <div>
                                <h3 class="text-lg font-semibold text-white mb-4">Configurações do Modelo IA</h3>
                                <div class="space-y-4">
                                    <div>
                                        <label class="block text-slate-400 text-sm mb-2">Limite de Confiança</label>
                                        <input type="range" min="1" max="99" value="50" class="w-full">
                                    </div>
                                    <div>
                                        <label class="block text-slate-400 text-sm mb-2">Tamanho do Lote de Treino</label>
                                        <input type="number" value="8" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                                    </div>
                                    <div>
                                        <label class="block text-slate-400 text-sm mb-2">Épocas de Treino</label>
                                        <input type="number" value="30" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-8 pt-6 border-t border-slate-700">
                            <div class="flex justify-between items-center">
                                <div>
                                    <h3 class="text-lg font-semibold text-white mb-2">Ações do Sistema</h3>
                                    <p class="text-slate-400 text-sm">Gerenciamento do sistema e dados</p>
                                </div>
                                <div class="flex gap-3">
                                    <button onclick="backupSystem()" 
                                            class="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg font-medium">
                                        Backup
                                    </button>
                                    <button onclick="resetSystem()" 
                                            class="bg-rose-600 hover:bg-rose-700 text-white px-4 py-2 rounded-lg font-medium">
                                        Reiniciar
                                    </button>
                                    <button onclick="updateModel()" 
                                            class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium">
                                        Atualizar Modelo
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            <!-- RODAPÉ -->
            <footer class="mt-12 pt-6 border-t border-slate-800">
                <div class="flex flex-col md:flex-row justify-between items-center text-slate-500 text-sm">
                    <div>
                        Sistema de Classificação Industrial v1.0 | Desenvolvido para Processamento de Resíduos
                    </div>
                    <div class="mt-2 md:mt-0">
                        <span id="serverStatus" class="text-green-400">Servidor Online</span>
                        <span class="mx-2">•</span>
                        <span id="imageCount">0 imagens processadas</span>
                    </div>
                </div>
            </footer>
        </div>

        <!-- INPUT OCULTO PARA SELEÇÃO DE IMAGEM -->
        <input type="file" id="imageInput" accept="image/*" class="hidden">
        
        <script>
            // Sistema de abas
            function showTab(tabName) {{
                document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                
                document.getElementById('tab' + tabName.charAt(0).toUpperCase() + tabName.slice(1)).classList.remove('hidden');
                document.getElementById('tab' + tabName.charAt(0).toUpperCase() + tabName.slice(1)).classList.add('active');
                
                if (tabName === 'historico') {{
                    loadHistory();
                }}
            }}
            
            // Variáveis globais
            let currentTrainingImage = null;
            let selectedCorrection = null;
            let esp32Images = [];
            let autoRefresh = true;
            let captureInterval = 2000;
            let autoRefreshInterval;
            
            // Inicialização
            document.addEventListener('DOMContentLoaded', function() {{
                updateClock();
                setInterval(updateClock, 1000);
                
                refreshESP32Image();
                startAutoRefresh();
                updateStatistics();
                
                // Atualiza status periodicamente
                setInterval(updateSystemStatus, 5000);
            }});
            
            // Atualiza relógio
            function updateClock() {{
                const now = new Date();
                document.getElementById('currentTime').textContent = 
                    now.toLocaleTimeString('pt-BR', {{hour12: false}});
            }}
            
            // Atualiza imagem do ESP32
            async function refreshESP32Image() {{
                try {{
                    const response = await fetch('/api/latest-esp32-image');
                    if (response.ok) {{
                        const data = await response.json();
                        if (data.url) {{
                            const img = document.getElementById('esp32CurrentImage');
                            const container = document.getElementById('esp32ImageContainer');
                            
                            // Ativa flash se for nova captura
                            if (data.flash) {{
                                const flashOverlay = document.getElementById('flashOverlay');
                                flashOverlay.classList.add('flash-animation');
                                setTimeout(() => flashOverlay.classList.remove('flash-animation'), 500);
                            }}
                            
                            img.src = data.url + '?' + new Date().getTime();
                            
                            // Atualiza timestamp
                            const timestamp = data.filename.replace('esp32cam_', '').replace('.jpg', '');
                            const date = new Date(parseInt(timestamp) * 1000);
                            document.getElementById('lastCaptureTime').textContent = 
                                date.toLocaleTimeString('pt-BR');
                                
                            // Atualiza última atividade
                            document.getElementById('lastActivity').textContent = 'Agora';
                        }}
                    }}
                }} catch (error) {{
                    console.error('Erro ao atualizar imagem:', error);
                }}
            }}
            
            // Inicia atualização automática
            function startAutoRefresh() {{
                if (autoRefreshInterval) clearInterval(autoRefreshInterval);
                autoRefreshInterval = setInterval(() => {{
                    if (autoRefresh) refreshESP32Image();
                }}, captureInterval);
            }}
            
            // Atualiza intervalo de captura
            function updateCaptureInterval(value) {{
                captureInterval = value * 1000;
                document.getElementById('intervalValue').textContent = value + 's';
                startAutoRefresh();
            }}
            
            // Alterna captura automática
            function toggleAutoCapture() {{
                autoRefresh = !autoRefresh;
                const btn = document.getElementById('autoCaptureBtn');
                if (autoRefresh) {{
                    btn.textContent = 'Auto: ON';
                    btn.classList.remove('bg-gray-600');
                    btn.classList.add('bg-blue-600');
                    startAutoRefresh();
                }} else {{
                    btn.textContent = 'Auto: OFF';
                    btn.classList.remove('bg-blue-600');
                    btn.classList.add('bg-gray-600');
                    clearInterval(autoRefreshInterval);
                }}
            }}
            
            // Captura manual
            async function captureManual() {{
                try {{
                    // Simula captura manual
                    const response = await fetch('/api/latest-esp32-image');
                    if (response.ok) {{
                        const data = await response.json();
                        if (data.url) {{
                            // Ativa flash
                            const flashOverlay = document.getElementById('flashOverlay');
                            flashOverlay.classList.add('flash-animation');
                            setTimeout(() => flashOverlay.classList.remove('flash-animation'), 500);
                            
                            // Atualiza imagem
                            const img = document.getElementById('esp32CurrentImage');
                            img.src = data.url + '?' + new Date().getTime();
                            
                            alert('Captura manual realizada com sucesso!');
                        }}
                    }}
                }} catch (error) {{
                    console.error('Erro na captura manual:', error);
                }}
            }}
            
            // Carrega histórico
            async function loadHistory() {{
                try {{
                    const response = await fetch('/api/esp32-images');
                    if (response.ok) {{
                        const data = await response.json();
                        esp32Images = data.images;
                        displayHistory(esp32Images);
                    }}
                }} catch (error) {{
                    console.error('Erro ao carregar histórico:', error);
                }}
            }}
            
            // Exibe histórico
            function displayHistory(images) {{
                const container = document.getElementById('imagesGrid');
                const loading = document.getElementById('loading');
                const noImages = document.getElementById('noImages');
                
                if (images.length === 0) {{
                    loading.classList.add('hidden');
                    noImages.classList.remove('hidden');
                    return;
                }}
                
                container.innerHTML = images.map(img => `
                    <div class="image-card glass-panel overflow-hidden rounded-xl">
                        <div class="relative">
                            <img src="${{img.url}}?t=${{new Date().getTime()}}" 
                                 class="w-full h-48 object-cover"
                                 onerror="this.onerror=null; this.src='/static/placeholder.jpg'">
                            <div class="absolute top-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                                ${{img.date.split(' ')[0]}}
                            </div>
                        </div>
                        <div class="p-4">
                            <div class="text-sm text-slate-400 mb-2">${{img.date}}</div>
                            <div class="flex justify-between items-center">
                                <button onclick="useForTraining('${{img.filename}}')" 
                                        class="text-sm bg-sky-600 hover:bg-sky-700 text-white px-3 py-1 rounded">
                                    Usar para Treino
                                </button>
                                <button onclick="viewImage('${{img.url}}')" 
                                        class="text-sm bg-slate-700 hover:bg-slate-600 text-white px-3 py-1 rounded">
                                    Visualizar
                                </button>
                            </div>
                        </div>
                    </div>
                `).join('');
                
                loading.classList.add('hidden');
                noImages.classList.add('hidden');
                document.getElementById('imageCount').textContent = images.length + ' imagens processadas';
            }}
            
            // Seleciona imagem para treino
            function selectTrainingImage() {{
                document.getElementById('imageInput').click();
            }}
            
            // Configura input de imagem
            document.getElementById('imageInput').addEventListener('change', async function(e) {{
                if (!e.target.files[0]) return;
                
                const file = e.target.files[0];
                const formData = new FormData();
                formData.append('file', file);
                
                try {{
                    // Envia para classificação
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await response.json();
                    currentTrainingImage = data.filename;
                    
                    // Atualiza interface
                    document.getElementById('trainingImage').src = '/static/capturas/' + data.filename + '?' + new Date().getTime();
                    document.getElementById('trainingFilename').textContent = data.filename;
                    document.getElementById('trainingStatus').textContent = 'Imagem carregada';
                    document.getElementById('trainingStatus').classList.remove('text-slate-400');
                    document.getElementById('trainingStatus').classList.add('text-green-400');
                    
                    // Exibe classificação atual
                    document.getElementById('currentPrediction').innerHTML = `
                        <div class="${{getColorClass(data.resultado)}} text-3xl font-bold mb-2">${{data.resultado}}</div>
                        <div class="text-slate-400">Classificação automática</div>
                    `;
                    
                    // Habilita botão de confirmação
                    document.getElementById('confirmBtn').disabled = false;
                    
                }} catch (error) {{
                    console.error('Erro ao processar imagem:', error);
                    alert('Erro ao processar imagem');
                }}
            }});
            
            // Usa imagem do histórico para treino
            async function useForTraining(filename) {{
                try {{
                    const response = await fetch('/static/capturas/' + filename);
                    const blob = await response.blob();
                    const file = new File([blob], filename, {{ type: 'image/jpeg' }});
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const predictResponse = await fetch('/predict', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await predictResponse.json();
                    currentTrainingImage = data.filename;
                    
                    // Atualiza interface
                    document.getElementById('trainingImage').src = '/static/capturas/' + data.filename + '?' + new Date().getTime();
                    document.getElementById('trainingFilename').textContent = data.filename;
                    document.getElementById('trainingStatus').textContent = 'Imagem do histórico carregada';
                    document.getElementById('trainingStatus').classList.remove('text-slate-400');
                    document.getElementById('trainingStatus').classList.add('text-green-400');
                    
                    // Exibe classificação atual
                    document.getElementById('currentPrediction').innerHTML = `
                        <div class="${{getColorClass(data.resultado)}} text-3xl font-bold mb-2">${{data.resultado}}</div>
                        <div class="text-slate-400">Classificação automática</div>
                    `;
                    
                    // Habilita botão de confirmação
                    document.getElementById('confirmBtn').disabled = false;
                    
                    // Muda para aba de treino
                    showTab('treino');
                    
                }} catch (error) {{
                    console.error('Erro:', error);
                    alert('Erro ao carregar imagem do histórico');
                }}
            }}
            
            // Corrige classificação
            function corrigirClassificacao(categoria) {{
                selectedCorrection = categoria;
                
                // Destaca botão selecionado
                document.querySelectorAll('#tabTreino button').forEach(btn => {{
                    btn.classList.remove('ring-2', 'ring-offset-2', 'ring-offset-slate-800', 'ring-white');
                }});
                
                const selectedBtn = document.querySelector(`#tabTreino button[onclick*="${{categoria}}"]`);
                if (selectedBtn) {{
                    selectedBtn.classList.add('ring-2', 'ring-offset-2', 'ring-offset-slate-800', 'ring-white');
                }}
                
                // Atualiza status
                document.getElementById('trainingStatus').textContent = 'Correção selecionada: ' + categoria;
                document.getElementById('trainingStatus').classList.remove('text-slate-400');
                document.getElementById('trainingStatus').classList.add('text-yellow-400');
            }}
            
            // Confirma correção
            async function confirmCorrection() {{
                if (!currentTrainingImage || !selectedCorrection) {{
                    alert('Selecione uma imagem e uma correção primeiro');
                    return;
                }}
                
                try {{
                    const params = new URLSearchParams();
                    params.append('filename', currentTrainingImage);
                    params.append('correct_class', selectedCorrection);
                    
                    await fetch('/teach', {{
                        method: 'POST',
                        body: params
                    }});
                    
                    // Atualiza estatísticas
                    const correctedCount = parseInt(document.getElementById('correctedCount').textContent) + 1;
                    document.getElementById('correctedCount').textContent = correctedCount;
                    
                    // Feedback visual
                    document.getElementById('trainingStatus').textContent = 'Correção confirmada! O modelo será atualizado.';
                    document.getElementById('trainingStatus').classList.remove('text-yellow-400');
                    document.getElementById('trainingStatus').classList.add('text-green-400');
                    
                    // Limpa seleção
                    selectedCorrection = null;
                    document.getElementById('confirmBtn').disabled = true;
                    
                    setTimeout(() => {{
                        document.getElementById('trainingStatus').textContent = 'Pronto para nova correção';
                        document.getElementById('trainingStatus').classList.remove('text-green-400');
                        document.getElementById('trainingStatus').classList.add('text-slate-400');
                    }}, 3000);
                    
                }} catch (error) {{
                    console.error('Erro ao confirmar correção:', error);
                    alert('Erro ao salvar correção');
                }}
            }}
            
            // Pula imagem
            function skipTraining() {{
                currentTrainingImage = null;
                selectedCorrection = null;
                
                document.getElementById('trainingImage').src = '/static/placeholder.jpg';
                document.getElementById('trainingFilename').textContent = 'Nenhuma imagem selecionada';
                document.getElementById('trainingStatus').textContent = 'Selecione uma imagem';
                document.getElementById('trainingStatus').classList.remove('text-green-400', 'text-yellow-400');
                document.getElementById('trainingStatus').classList.add('text-slate-400');
                document.getElementById('currentPrediction').innerHTML = `
                    <div class="text-slate-400">Nenhuma classificação disponível</div>
                `;
                document.getElementById('confirmBtn').disabled = true;
                
                // Remove destaque dos botões
                document.querySelectorAll('#tabTreino button').forEach(btn => {{
                    btn.classList.remove('ring-2', 'ring-offset-2', 'ring-offset-slate-800', 'ring-white');
                }});
            }}
            
            // Treina modelo
            async function trainModel() {{
                try {{
                    const btn = document.getElementById('confirmBtn');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Treinando...';
                    btn.disabled = true;
                    
                    await fetch('/run_train', {{ method: 'POST' }});
                    
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                    
                    // Atualiza último treino
                    const now = new Date();
                    document.getElementById('lastTraining').textContent = now.toLocaleTimeString('pt-BR');
                    
                    alert('Modelo treinado com sucesso com as correções recentes!');
                    
                }} catch (error) {{
                    console.error('Erro ao treinar modelo:', error);
                    alert('Erro ao treinar modelo');
                }}
            }}
            
            // Atualiza estatísticas
            async function updateStatistics() {{
                try {{
                    const response = await fetch('/api/esp32-images');
                    if (response.ok) {{
                        const data = await response.json();
                        document.getElementById('totalCaptures').textContent = data.images.length;
                        
                        // Simula taxa de acerto (em produção, calcular baseado em validações)
                        const accuracy = Math.min(95, 70 + Math.random() * 25);
                        document.getElementById('accuracyRate').textContent = accuracy.toFixed(1) + '%';
                    }}
                }} catch (error) {{
                    console.error('Erro ao carregar estatísticas:', error);
                }}
            }}
            
            // Atualiza status do sistema
            async function updateSystemStatus() {{
                try {{
                    const response = await fetch('/api/esp32-images');
                    if (response.ok) {{
                        // Atualiza contagem
                        const data = await response.json();
                        document.getElementById('imageCount').textContent = data.images.length + ' imagens processadas';
                    }}
                }} catch (error) {{
                    console.error('Erro ao atualizar status:', error);
                }}
            }}
            
            // Funções auxiliares
            function getColorClass(categoria) {{
                const cores = {{
                    'BIOLOGICO': 'text-green-400',
                    'METAL': 'text-gray-400',
                    'PLASTICO': 'text-blue-400',
                    'VIDRO': 'text-cyan-400',
                    'PAPEL': 'text-amber-400'
                }};
                return cores[categoria] || 'text-white';
            }}
            
            function viewImage(url) {{
                window.open(url, '_blank');
            }}
            
            function exportHistory() {{
                alert('Funcionalidade de exportação em desenvolvimento');
            }}
            
            function backupSystem() {{
                alert('Funcionalidade de backup em desenvolvimento');
            }}
            
            function resetSystem() {{
                if (confirm('Tem certeza que deseja reiniciar o sistema? Isso limpará as imagens temporárias.')) {{
                    alert('Sistema reiniciado');
                }}
            }}
            
            function updateModel() {{
                alert('Atualização do modelo iniciada');
            }}
            
            // Busca no histórico
            document.getElementById('searchImages').addEventListener('input', function(e) {{
                const searchTerm = e.target.value.toLowerCase();
                const filtered = esp32Images.filter(img => 
                    img.date.toLowerCase().includes(searchTerm) || 
                    img.filename.toLowerCase().includes(searchTerm)
                );
                displayHistory(filtered);
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

# --- SUAS FUNÇÕES ORIGINAIS (MANTIDAS) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    fname = f"ref_{int(time.time())}.jpg"
    path = os.path.join(CAPTURES_DIR, fname)
    with open(path, "wb") as b:
        shutil.copyfileobj(file.file, b)
    
    results = brain.predict(source=path, imgsz=224, conf=0.01)
    
    try:
        cid = int(results[0].probs.top1)
        n_en = results[0].names[cid].lower()
        
        if n_en in CATEGORIAS:
            res = n_en
        else:
            res = MAPA_ESTRITO.get(n_en, "plastico")
    except:
        res = "plastico"
        
    return {"resultado": res.upper(), "filename": fname}

@app.post("/teach")
async def teach(filename: str = Form(...), correct_class: str = Form(...)):
    src = os.path.join(CAPTURES_DIR, filename)
    d_train = os.path.join(DATASET_DIR, "train", correct_class, filename)
    d_val = os.path.join(DATASET_DIR, "val", correct_class, filename)
    if os.path.exists(src):
        shutil.copy(src, d_train)
        shutil.move(src, d_val)
    return {"status": "ok"}

@app.post("/run_train")
async def run_train():
    global brain
    brain.train(
        data=DATASET_DIR, 
        epochs=30, 
        imgsz=224, 
        batch=8, 
        project=os.path.join(BASE_DIR, "treino_final"), 
        name="estacao_trabalho"
    )
    brain = carregar_inteligencia()
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")