# ⚖️ Auditoria Interseccional de Viés Racial (TCC)

> **Projeto:** Mitigação de Viés Racial em Algoritmos de Reconhecimento Facial  
> **Instituição:** Instituto Federal da Bahia (IFBA) - Campus Salvador  
> **Curso:** Análise e Desenvolvimento de Sistemas (ADS)  
> **Autor:** Marcílio Motta  
> **Orientadora:** Profa. Flávia  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![Status](https://img.shields.io/badge/Status-Fase%201%20(Auditoria)-yellow?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📄 Resumo do Projeto

Este projeto visa auditar, quantificar e mitigar disparidades de desempenho (viés algorítmico) em sistemas de reconhecimento facial de código aberto. 

Diferente de abordagens tradicionais, este trabalho utiliza uma **metodologia interseccional**, avaliando o comportamento de modelos clássicos (`Dlib/HOG`) e modernos baseados em Deep Learning (`DeepFace/VGG`) através do cruzamento de variáveis demográficas de **Raça e Gênero**. O objetivo é estabelecer uma linha de base técnica robusta para a futura mitigação de discriminação algorítmica.

---

## ⚙️ Arquitetura e Engenharia de Dados

O sistema opera através de um pipeline estruturado em três etapas para garantir a confiabilidade dos resultados:

### 1. Aquisição e Curadoria (Data Curation)
Utilizamos um script de *web scraping* (`download_tcc.py`) para a coleta massiva de imagens candidatas, seguido de um rigoroso protocolo de seleção.
* **Metodologia de Qualidade (Human-in-the-Loop):** Diferente de datasets puramente automáticos, este projeto aplica uma etapa de **validação manual** em todas as amostras. Isso assegura que 100% das imagens do dataset final contenham:
    * Integridade visual (sem artefatos ou baixa resolução).
    * Presença de **apenas um rosto** (exclusão de fotos em grupo).
    * Identidade correta (garantia de *Ground Truth*).

### 2. Estrutura de Dados Interseccional
O dataset é organizado hierarquicamente para permitir a análise cruzada de viés:
```text
dataset_auditoria/
├── Negros_Homens/       # [Grupo Demográfico]
│   └── Idris_Elba/      # [Identidade] -> Fotos.jpg
├── Negros_Mulheres/
├── Brancos_Homens/
└── Brancos_Mulheres/
``` 

### 3. Motor de Auditoria (Benchmarking Strategy)
O script principal (`main.py`) implementa o padrão de projeto *Strategy* para submeter o dataset a múltiplos modelos simultaneamente. Ele executa milhares de comparações de pares (1:1) para calcular:
* **Pares Genuínos:** Comparações entre fotos da mesma identidade (Esperado: Match).
* **Pares Impostores:** Comparações entre identidades diferentes do mesmo grupo (Esperado: Mismatch).

---

## 📊 Métricas de Avaliação

Para garantir rigor acadêmico, o sistema calcula as seguintes métricas por grupo demográfico:

| Métrica | Definição | Interpretação no TCC |
| :--- | :--- | :--- |
| **TPR (True Positive Rate)** | Taxa de Verdadeiros Positivos | **Sensibilidade:** A capacidade do modelo de reconhecer corretamente um indivíduo do grupo. |
| **FPR (False Positive Rate)** | Taxa de Falsos Positivos | **Viés de Segurança:** A frequência com que o modelo confunde duas pessoas diferentes. Um FPR alto em grupos minoritários indica alto risco de discriminação. |

---

## 🛠️ Pré-requisitos e Instalação

Este projeto utiliza bibliotecas de alta performance (C++ bindings) que requerem configuração específica no Windows.

### 1. Dependências de Sistema (Obrigatório)
Antes de instalar o Python, garanta que possui:
* **Visual Studio Build Tools:** Instale a carga de trabalho *"Desenvolvimento para desktop com C++"* (Necessário para compilar o `dlib` e `CMake`).
* **CMake:** Ferramenta de build (`pip install cmake`).

### 2. Instalação do Projeto

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/marcilio-motta/fr-bias-mitigation.git
    cd fr-bias-mitigation
    ```

2.  **Configure o Ambiente Virtual:**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instale as Dependências:**
    ```powershell
    pip install -r requirements.txt
    ```

---

## 🚀 Como Executar

### Pré-requisito: Dataset
O repositório já inclui o diretório `dataset_auditoria` pré-processado e curado manualmente para garantir a reprodutibilidade dos resultados apresentados no TCC.

*(Opcional) Caso deseje recriar o dataset do zero, execute `python download_tcc.py`. Note que isso pode trazer novas imagens da web que exigirão limpeza manual.*

### Passo 1: Execução da Auditoria
Com o ambiente virtual ativado, execute o script principal. Ele lerá as imagens da pasta `dataset_auditoria` e comparará os modelos:

```powershell
python main.py
```  

### Passo 2: Análise de Resultados
O terminal exibirá um relatório formatado comparando a performance do Modelo Clássico vs. Modelo Moderno.

Atenção: Compare o FPR entre o grupo Negros_Mulheres e Brancos_Homens. A diferença entre esses números é a prova estatística do viés.

## 📅 Roadmap do TCC
- [x] Fase 1: Auditoria (Baseline) - Diagnóstico e quantificação do viés.

- [ ] Fase 2: Mitigação - Fine-tuning de modelos (FaceNet/ResNet) com datasets balanceados.

- [ ] Fase 3: Validação - Re-auditoria para comprovar a redução da disparidade.

Desenvolvido com 💙 e Python por Marcílio Motta.