#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include "base64.h"

// --- NOVAS CREDENCIAIS ---
const char* ssid = "Cisco09968";
const char* password = "arrogantes321@2026";

const char* websocket_server = "192.168.1.107";
const uint16_t websocket_port = 7860;
const char* websocket_path = "/ws/esp32cam";

WebSocketsClient webSocket;

// Configuração da Câmera (Modelo AI Thinker)
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
unsigned long captureInterval = 5000; // Padrão lento (5s)
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
  config.frame_size = FRAMESIZE_VGA; // 640x480
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Falha Câmera: 0x%x\n", err);
    return;
  }
  Serial.println("Câmera inicializada com sucesso.");
  cameraInitialized = true;
}

String captureAndEncodeImage() {
  if (!cameraInitialized) return "";
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) return "";
  
  Serial.printf("Imagem capturada: %d bytes\n", fb->len);
  
  String imageBase64 = base64::encode(fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return imageBase64;
}

void handleWebSocketText(uint8_t * payload, size_t length) {
  DynamicJsonDocument doc(256);
  DeserializationError error = deserializeJson(doc, payload);
  
  if (!error) {
    const char* mode = doc["mode"];
    if (mode) {
      if (strcmp(mode, "fast") == 0) {
        captureInterval = 1000; // 1 segundo
        Serial.println("MUDANÇA: Modo RÁPIDO ativado (1s)");
      } else if (strcmp(mode, "slow") == 0) {
        captureInterval = 5000; // 5 segundos
        Serial.println("MUDANÇA: Modo LENTO ativado (5s)");
      }
    }
  }
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Desconectado. Tentando reconectar...");
      break;
    case WStype_CONNECTED:
      Serial.println("[WS] Conectado ao Servidor!");
      break;
    case WStype_TEXT:
      handleWebSocketText(payload, length);
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  WiFi.setAutoReconnect(true); // Tenta reconectar se cair
  WiFi.begin(ssid, password);
  
  Serial.print("Conectando ao WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nWiFi Conectado!");
  Serial.print("IP Cam: "); Serial.println(WiFi.localIP());

  setupCamera();
  
  webSocket.begin(websocket_server, websocket_port, websocket_path);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(3000);
}

void loop() {
  webSocket.loop();
  unsigned long currentTime = millis();
  
  if (currentTime - lastCaptureTime >= captureInterval) {
    lastCaptureTime = currentTime;
    
    if (webSocket.isConnected() && cameraInitialized) {
      String imageBase64 = captureAndEncodeImage();
      if (imageBase64.length() > 0) {
        // Envia a imagem
        webSocket.sendTXT(imageBase64);
        Serial.println("Imagem enviada para análise.");
      }
    }
  }
}