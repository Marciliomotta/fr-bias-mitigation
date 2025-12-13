# ⚖️ FR Bias Mitigation (TCC)

> **Mitigação de Viés Racial em Algoritmos de Reconhecimento Facial**
> *Trabalho de Conclusão de Curso (TCC) - Instituto Federal da Bahia (IFBA)*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 📄 Sobre o Projeto

Este repositório contém a implementação prática do meu Trabalho de Conclusão de Curso (TCC). O objetivo central é desenvolver uma metodologia técnica para **auditar, mitigar e validar** a redução de viés racial em sistemas de reconhecimento facial de código aberto.

O projeto ataca o problema da disparidade de acurácia entre grupos demográficos (ex: rostos negros vs. brancos) em modelos populares como `dlib` e `FaceNet`.

### 🎯 Objetivos
- **Auditar:** Quantificar o viés em modelos pré-treinados (Baseline).
- **Mitigar:** Aplicar técnicas de *fine-tuning* com datasets demograficamente balanceados.
- **Validar:** Comprovar a redução do viés através de uma API de inferência comparativa.

---

## 🚀 Metodologia & Pipeline

O desenvolvimento está dividido em 4 fases práticas:

1.  **Auditoria (Baseline):** Scripts para medir Taxas de Falso Positivo (FPR) e Verdadeiro Positivo (TPR) em modelos *off-the-shelf* (dlib, DeepFace).
2.  **Mitigação (Fine-Tuning):** Re-treinamento de modelos (ex: FaceNet/ResNet) utilizando PyTorch/TensorFlow e datasets equitativos.
3.  **Implementação (API):** Desenvolvimento de uma API RESTful com **FastAPI** para servir o modelo mitigado.
4.  **Validação Final:** Re-execução dos scripts de auditoria contra a API para gerar métricas "Antes vs. Depois".
