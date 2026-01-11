import os
from .interfaces import IFaceModel

class BiasAuditor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def run_audit(self, model: IFaceModel):
        results = {}
        grupos = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        for grupo in grupos:
            path_grupo = os.path.join(self.dataset_path, grupo)
            identidades = {}
            pessoas = [p for p in os.listdir(path_grupo) if os.path.isdir(os.path.join(path_grupo, p))]
            
            for pessoa in pessoas:
                path_pessoa = os.path.join(path_grupo, pessoa)
                identidades[pessoa] = []
                for img in os.listdir(path_pessoa):
                    desc = model.get_descriptors(os.path.join(path_pessoa, img))
                    if desc: identidades[pessoa].append(desc[0])

            results[grupo] = self._calculate_metrics(model, identidades)
        return results

    def _calculate_metrics(self, model, identidades):
        tp, tot_p, fp, tot_n = 0, 0, 0, 0
        nomes = list(identidades.keys())
        
        for p, vecs in identidades.items():
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    tot_p += 1
                    if model.compare(vecs[i], vecs[j]): tp += 1

        for i in range(len(nomes)):
            for j in range(i+1, len(nomes)):
                for v_a in identidades[nomes[i]]:
                    for v_b in identidades[nomes[j]]:
                        tot_n += 1
                        if model.compare(v_a, v_b): fp += 1

        return {"TPR": (tp/tot_p*100) if tot_p > 0 else 0, "FPR": (fp/tot_n*100) if tot_n > 0 else 0}