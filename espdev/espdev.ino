#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// --- NOVAS CREDENCIAIS ---
const char* ssid = "Cisco09968";
const char* password = "arrogantes321@2026";

const char* websocket_server = "192.168.1.107";
const uint16_t websocket_port = 7860;
const char* websocket_path = "/ws/esp32dev"; // Caminho do ESP32 DEV

WebSocketsClient webSocket;

// --- PINOS ---
#define TRIG_PIN 5
#define ECHO_PIN 18

#define SERVO1_PIN 13
#define SERVO2_PIN 14 
#define SERVO3_PIN 15 

Servo servo1;
Servo servo2;
Servo servo3;

unsigned long lastSensorCheck = 0;
const unsigned long sensorInterval = 200; 
bool objectDetected = false;
bool servosMoving = false;

void setup() {
  Serial.begin(115200);
  Serial.println("Iniciando Sistema ESP32 Dev...");

  // Servos
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo3.attach(SERVO3_PIN);
  resetServos();

  // Sensor
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // WiFi
  WiFi.setAutoReconnect(true);
  WiFi.begin(ssid, password);
  Serial.print("Conectando WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nWiFi OK");
  Serial.print("IP Dev: "); Serial.println(WiFi.localIP());

  // WebSocket
  webSocket.begin(websocket_server, websocket_port, websocket_path);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop() {
  webSocket.loop();

  unsigned long currentMillis = millis();
  if (!servosMoving && currentMillis - lastSensorCheck >= sensorInterval) {
    lastSensorCheck = currentMillis;
    checkUltrasonic();
  }
}

void checkUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  
  if (duration == 0) {
    if (objectDetected) sendSensorStatus(false);
    return;
  }

  float distance = (duration * 0.0343) / 2;
  
  // Range de 7 a 13 cm
  bool currentlyDetected = (distance >= 7 && distance <= 13);

  if (currentlyDetected != objectDetected) {
    objectDetected = currentlyDetected;
    sendSensorStatus(objectDetected);
    
    if (objectDetected) Serial.println(">> OBJETO DETECTADO! Iniciando capturas...");
    else Serial.println(">> OBJETO REMOVIDO. Parando capturas.");
  }
}

void sendSensorStatus(bool detected) {
  DynamicJsonDocument doc(256);
  doc["sensor_active"] = detected;
  
  String jsonString;
  serializeJson(doc, jsonString);
  webSocket.sendTXT(jsonString);
}

void activateServosSequence() {
  Serial.println(">>> ATIVANDO SERVOS (Confirmação Visual)");
  servosMoving = true;

  servo1.write(180);
  servo2.write(180);
  servo3.write(180);
  
  delay(1000);
  
  resetServos();
  
  delay(500);
  servosMoving = false;
  Serial.println(">>> SERVOS FINALIZADOS");
}

void resetServos() {
  servo1.write(0);
  servo2.write(0);
  servo3.write(0);
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Desconectado");
      break;
    case WStype_CONNECTED:
      Serial.println("[WS] Conectado ao Backend");
      break;
    case WStype_TEXT:
      Serial.printf("[WS] Dados recebidos: %s\n", payload);
      
      DynamicJsonDocument doc(1024);
      DeserializationError error = deserializeJson(doc, payload);
      
      if (!error) {
        const char* target = doc["target"];
        const char* result_class = doc["result_class"];
        
        // Verifica se o servidor enviou o resultado da IA
        if (target && strcmp(target, "dev") == 0 && result_class) {
          Serial.println("============================================");
          Serial.print("  RESULTADO DA IA: ");
          Serial.println(result_class);
          Serial.println("============================================");
          
          // Aqui você pode decidir acionar servos baseado na classe
          // Por enquanto, ativa a sequência padrão
          activateServosSequence();
        }
      }
      break;
  }
}