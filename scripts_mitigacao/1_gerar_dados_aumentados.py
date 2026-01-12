import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# CONFIGURACOES
ORIGEM_DIR = "dataset_auditoria"
DESTINO_BASE = "dataset_treino"
QTD_VARIACOES = 50 

# Configuracao da Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

def criar_dataset_aumentado():
    print(f"Iniciando fabrica de dados... Meta: {QTD_VARIACOES} variacoes por foto.")
    
    if os.path.exists(DESTINO_BASE):
        shutil.rmtree(DESTINO_BASE)
    
    grupos = [d for d in os.listdir(ORIGEM_DIR) if os.path.isdir(os.path.join(ORIGEM_DIR, d))]
    total_gerado = 0
    
    for grupo in grupos:
        path_grupo = os.path.join(ORIGEM_DIR, grupo)
        pessoas = [p for p in os.listdir(path_grupo) if os.path.isdir(os.path.join(path_grupo, p))]
        
        for pessoa in pessoas:
            path_pessoa_origem = os.path.join(path_grupo, pessoa)
            fotos = [f for f in os.listdir(path_pessoa_origem) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            fotos.sort() # Garante ordem deterministica
            
            print(f"   Processando: {pessoa} ({len(fotos)} originais)...")
                        
            for i, foto in enumerate(fotos):
                img_path = os.path.join(path_pessoa_origem, foto)
                
                try:
                    img = load_img(img_path)
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)
                except:
                    print(f"      Erro ao ler {foto}, pulando.")
                    continue

                # Se for a primeira foto da lista, vai para validacao. O resto para treino.
                if i == 0:
                    destino_split = "val"
                else:
                    destino_split = "train"
                
                path_pessoa_destino = os.path.join(DESTINO_BASE, destino_split, pessoa)
                os.makedirs(path_pessoa_destino, exist_ok=True)
                
                j = 0
                for batch in datagen.flow(x, batch_size=1, 
                                          save_to_dir=path_pessoa_destino, 
                                          save_prefix='aug', 
                                          save_format='jpg'):
                    j += 1
                    total_gerado += 1
                    if j >= QTD_VARIACOES:
                        break
                        
    print(f"\nProcesso Finalizado!")
    print(f"Total de novas imagens geradas: {total_gerado}")

if __name__ == "__main__":
    criar_dataset_aumentado()