import os

def criar_ambiente_fase2():
    pastas_novas = [
        "dataset_treino/raw",           
        "dataset_treino/augmented",     
        "dataset_treino/train",        
        "dataset_treino/val",           
        "modelos_salvos",               
        "scripts_mitigacao",            
        "logs_treinamento"             
    ]
    
    print("Construindo laboratório da Fase 2...")
    
    for pasta in pastas_novas:
        os.makedirs(pasta, exist_ok=True)
        with open(os.path.join(pasta, ".gitkeep"), "w") as f:
            pass
        print(f"Criada: {pasta}/")

    print("\n Ambiente pronto!")

if __name__ == "__main__":
    criar_ambiente_fase2()