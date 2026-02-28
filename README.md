 # Sistema de Classificação de Resíduos com IA

Sistema de visão computacional para classificação automática de resíduos industriais, utilizando inteligência artificial e processamento em tempo real.

## 🎯 Objetivo

Automatizar a identificação e separação de resíduos, contribuindo para processos industriais mais eficientes e sustentáveis.

### Categorias classificadas

* Biológico
* Metal
* Plástico
* Vidro
* Papel

---

## 🔧 Tecnologias Utilizadas

* Python — backend e processamento
* FastAPI — API de alta performance
* YOLO / Ultralytics — detecção e classificação por visão computacional
* ESP32-CAM — captura de imagens e integração embarcada
* ESP32 — Controle dos atuadores(servo-motores) e do funcionamento da esteira
* Docker — containerização e portabilidade

---

## 📁 Estrutura do Projeto

```
backend/
 ├── app.py            # API principal
 ├── dataset/          # Dados de treino
 ├── weights/          # Pesos do modelo treinado
 ├── runs/             # Resultados de treino
 ├── static/           # Arquivos estáticos
```

---

## 🚀 Como Executar

### 1️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Iniciar a API

```bash
python backend/app.py
```

A API estará disponível em:

```
http://localhost:8000
```

---

## 🧠 Funcionalidades

* Classificação automática de resíduos em tempo real
* Integração com câmera/ESP32
* Estrutura pronta para treinamento contínuo
* Arquitetura modular e escalável

---

## 📌 Status do Projeto

🚧 Em desenvolvimento — melhorias contínuas em precisão e desempenho.

---

## 📜 Licença

Este projeto está sob a licença MIT.

