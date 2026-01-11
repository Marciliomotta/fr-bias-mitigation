import os
from src.models import DeepFaceAdapter, FaceRecognitionAdapter 
from src.auditor import BiasAuditor

DATASET_DIR = os.path.join(os.getcwd(), "dataset_auditoria")

def main():
    modelos = [FaceRecognitionAdapter(), DeepFaceAdapter()]
    auditor = BiasAuditor(DATASET_DIR)
    global_results = {m.get_name(): auditor.run_audit(m) for m in modelos}

    print("\n" + "#"*65 + f"\n{' RELATÓRIO COMPARATIVO FINAL DE VIÉS':^65}\n" + "#"*65)
    for grupo in sorted(list(global_results.values())[0].keys()):
        print(f"\n GRUPO: {grupo}\n{'MODELO':<30} | {'ACERTO (TPR)':<15} | {'ERRO (FPR)':<15}\n" + "-"*65)
        for mod, res in global_results.items():
            print(f"{mod:<30} | {res[grupo]['TPR']:6.2f}%         | {res[grupo]['FPR']:6.2f}%")

if __name__ == "__main__": main()