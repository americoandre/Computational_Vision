                                                    Sistema de Classificação de Resíduos com  Visão Computacional  Edge AI
                                                    
Este projeto consiste em um sistema de visão computacional voltado para a classificação automática
de resíduos industriais. Através da integração entre inteligência artificial e hardware embarcado,
a solução permite a identificação e separação de materiais em tempo real, otimizando processos de 
reciclagem e gestão ambiental.

            Propósito
O objetivo central é automatizar a triagem de materiais descartados, reduzindo a margem de erro humana
e aumentando a velocidade de processamento em linhas de separação. O sistema atua diretamente na
identificação de cinco categorias principais:

Biológico
Metal
Plástico
Vidro
Papel

          Tecnologias e Ferramentas
A arquitetura do sistema foi desenhada para equilibrar performance computacional e viabilidade física:
Linguagem e Frameworks: Python para a lógica de backend e FastAPI para a disponibilização de 
uma API de alta performance.
Visão Computacional: Utilização do YOLO (Ultralytics) para detecção e classificação de objetos.
Hardware e Embarcados: ESP32-CAM para captura de imagens e ESP32 padrão para o controle físico da 
esteira e dos servo-motores (atuadores).
Infraestrutura: Docker para garantir a portabilidade e facilitar o deploy em 
diferentes ambientes industriais.

                                                                 Estrutura do Repositório
                                       
O projeto está organizado de forma modular para facilitar a manutenção e o treinamento do modelo:

Plaintext
backend/
 ├── app.py            # Ponto de entrada da API e rotas principais
 ├── dataset/          # Base de dados utilizada para o treinamento da IA
 ├── weights/          # Arquivos de pesos (.pt) do modelo treinado
 ├── runs/             # Logs e métricas resultantes dos ciclos de treino
 ├── static/           # Arquivos estáticos e recursos de suporte
 
                                                               Procedimentos de Execução
Instalação de Dependências
Certifique-se de ter o Python instalado e execute o comando abaixo para configurar o ambiente:

 Bash
pip install -r requirements.txt
Inicialização do Sistema
Para colocar a API em funcionamento, execute o script principal:

 Bash
python backend/app.py
O serviço será instanciado localmente no endereço: http://localhost:8000

                                                               Capacidades do Sistema
                                                               
Processamento em Tempo Real: Classificação instantânea assim que o resíduo é detectado pela câmera.
Integração de Hardware: Comunicação direta entre o processamento de imagem e o controle físico dos motores.
Aprendizado Contínuo: Estrutura preparada para a inclusão de novos dados e refinamento da precisão do modelo.
Escalabilidade: Arquitetura modular que permite a expansão para novas categorias de resíduos ou múltiplas linhas de produção.

                                                                Status de Desenvolvimento
O sistema encontra-se em fase de aprimoramento. O foco atual reside no aumento da precisão de detecção
em condições variadas de iluminação e na redução da latência entre a captura da imagem e a resposta do atuador.
