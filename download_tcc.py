import os
import shutil
import concurrent.futures
import time
from bing_image_downloader import downloader

DATASET_DIR = "dataset_auditoria"
QTD_DOWNLOAD = 20  
MAX_THREADS = 4 

structure = {
    "Negros_Homens": ["Lazaro Ramos", "Idris Elba", "Seu Jorge", "Michael B Jordan", "Denzel Washington"],
    "Negros_Mulheres": ["Tais Araujo", "Viola Davis", "Iza Cantora", "Lupita Nyong'o", "Zendaya"],
    "Brancos_Homens": ["Rodrigo Santoro", "Brad Pitt", "Wagner Moura", "Chris Evans", "Tom Cruise"],
    "Brancos_Mulheres": ["Marina Ruy Barbosa", "Nicole Kidman", "Xuxa Meneghel", "Scarlett Johansson", "Margot Robbie"]
}

def processar_grupo(args):
    grupo_nome, lista_pessoas = args
    path_grupo = os.path.join(DATASET_DIR, grupo_nome)
    
    for pessoa in lista_pessoas:
        print(f" Baixando imagens RAW de: {pessoa}...")
        
        try:
            downloader.download(
                f"{pessoa} face", 
                limit=QTD_DOWNLOAD, 
                output_dir="temp_download", 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=5,
                verbose=False
            )
        except Exception as e:
            print(f"Erro ao baixar {pessoa}: {e}")
            continue 
        
        nome_formatado = pessoa.replace(" ", "_")
        path_pessoa = os.path.join(path_grupo, nome_formatado)
        os.makedirs(path_pessoa, exist_ok=True)
        
        pasta_temp_bing = os.path.join("temp_download", f"{pessoa} face")
        
        if os.path.exists(pasta_temp_bing):
            arquivos = os.listdir(pasta_temp_bing)
            for i, arquivo in enumerate(arquivos):
                origem = os.path.join(pasta_temp_bing, arquivo)
                extensao = os.path.splitext(arquivo)[1].lower()
                
                if extensao in ['.jpg', '.jpeg', '.png']:
                    novo_nome = f"{nome_formatado}_{i+1:02d}{extensao}"
                    destino = os.path.join(path_pessoa, novo_nome)
                    try:
                        shutil.move(origem, destino)
                    except:
                        pass
            
            shutil.rmtree(pasta_temp_bing, ignore_errors=True)

def main():
    if os.path.exists("temp_download"): shutil.rmtree("temp_download", ignore_errors=True)
    
    print(" Iniciando Coleta SEM FILTRO (Modo Rápido)...")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(processar_grupo, item) for item in list(structure.items())]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Uma thread falhou: {exc}')

    finally:
        print("Realizando limpeza final...")
        time.sleep(2) 
        if os.path.exists("temp_download"): 
            shutil.rmtree("temp_download", ignore_errors=True)
        
        print("\n Download concluído! Agora vá nas pastas e APAGUE as fotos ruins manualmente.")

if __name__ == "__main__":
    main()