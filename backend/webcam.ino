#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include "base64.h"

// Configurações WiFi
const char* ssid = "Cisco09968";
const char* password = "arrogantes321@2026";

// Configurações WebSocket
const char* websocket_server = "192.168.1.113";  // Ex: "192.168.1.100"
const uint16_t websocket_port = 8000;
const char* websocket_path = "/ws/esp32cam";

WebSocketsClient webSocket;

// Configuração da câmera
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

unsigned long lastCaptureTime = 0;
const unsigned long captureInterval = 2000; // 2 segundos
bool cameraInitialized = false;

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Reduza a resolução se tiver problemas de memória
  // FRAMESIZE_ + QVGA|CIF|VGA|SVGA|XGA|SXGA|UXGA
  config.frame_size = FRAMESIZE_VGA; // 640x480 (menor que SVGA)
  config.jpeg_quality = 15; // 0-63 (menor = melhor qualidade)
  config.fb_count = 1;

  // Inicializa a câmera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("FALHA: Inicialização da câmera (0x%x)\n", err);
    cameraInitialized = false;
    return;
  }
  
  Serial.println("SUCESSO: Câmera inicializada");
  cameraInitialized = true;
}

String captureAndEncodeImage() {
  if (!cameraInitialized) {
    Serial.println("ERRO: Câmera não inicializada");
    return "";
  }
  
  // Captura frame
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("ERRO: Falha ao capturar imagem");
    return "";
  }

  Serial.printf("Imagem capturada: %d bytes\n", fb->len);
  
  // Converte para base64
  String imageBase64 = base64::encode(fb->buf, fb->len);
  
  // Libera buffer
  esp_camera_fb_return(fb);
  
  return imageBase64;
}

void handleWebSocketText(uint8_t * payload, size_t length) {
  Serial.print("Resposta do servidor: ");
  Serial.println((char*)payload);
  
  // Processa resposta JSON
  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, payload);
  
  if (error) {
    Serial.print("ERRO JSON: ");
    Serial.println(error.c_str());
    return;
  }
  
  const char* status = doc["status"];
  const char* result = doc["result"];
  
  if (status && strcmp(status, "processed") == 0) {
    Serial.print("Classificação: ");
    Serial.println(result);
    
    // Aqui você pode adicionar ações baseadas na classificação
    // Ex: acionar LEDs, motores, etc.
  } else if (status && strcmp(status, "error") == 0) {
    const char* message = doc["message"];
    Serial.print("ERRO do servidor: ");
    Serial.println(message);
  }
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("Desconectado do WebSocket");
      break;
      
    case WStype_CONNECTED:
      Serial.println("Conectado ao servidor WebSocket");
      break;
      
    case WStype_TEXT:
      handleWebSocketText(payload, length);
      break;
      
    case WStype_ERROR:
      Serial.println("Erro no WebSocket");
      break;
      
    case WStype_PING:
      Serial.println("Ping recebido");
      break;
      
    case WStype_PONG:
      Serial.println("Pong recebido");
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000); // Aguarda estabilização
  
  Serial.println("\n\n=================================");
  Serial.println("   ESP32-CAM Vision System");
  Serial.println("=================================\n");

  // Conecta ao WiFi
  WiFi.begin(ssid, password);
  Serial.print("Conectando ao WiFi");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nSUCESSO: Conectado ao WiFi!");
    Serial.print("IP Local: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFALHA: Não conseguiu conectar ao WiFi");
    Serial.println("Reiniciando em 5 segundos...");
    delay(5000);
    ESP.restart();
  }

  // Inicializa câmera
  setupCamera();
  
  if (!cameraInitialized) {
    Serial.println("AVISO: Sistema operando sem câmera");
  }

  // Configura WebSocket
  webSocket.begin(websocket_server, websocket_port, websocket_path);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(3000);
  
  Serial.println("Sistema inicializado e pronto!");
}

void loop() {
  webSocket.loop();
  
  unsigned long currentTime = millis();
  
  // Captura e envia imagem a cada 2 segundos
  if (currentTime - lastCaptureTime >= captureInterval) {
    lastCaptureTime = currentTime;
    
    if (webSocket.isConnected() && cameraInitialized) {
      Serial.println("\n--- Iniciando captura ---");
      
      String imageBase64 = captureAndEncodeImage();
      
      if (imageBase64.length() > 0) {
        Serial.printf("Imagem codificada: %d caracteres\n", imageBase64.length());
        
        // Envia imagem via WebSocket
        if (webSocket.sendTXT(imageBase64)) {
          Serial.println("Imagem enviada com sucesso");
        } else {
          Serial.println("Falha ao enviar imagem");
        }
      } else {
        Serial.println("Não foi possível capturar imagem");
      }
    } else {
      if (!webSocket.isConnected()) {
        Serial.println("WebSocket desconectado. Tentando reconectar...");
      }
      if (!cameraInitialized) {
        Serial.println("Câmera não disponível");
      }
    }
  }
  
  delay(50); // Pequeno delay para não sobrecarregar
}