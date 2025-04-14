import numpy as np

from base import Classifier


class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """

        self.cl_bin = cl_bin 
        self.classifieurs = []

        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """      
        self.classifiers = []
        for c in np.unique(label_set):
            cl = copy.deepcopy(self.cl_bin)
            labels = np.where(label_set == c, 1, -1)
            cl.train(desc_set, labels)
            self.classifiers.append(cl)

        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        scores = [clf.score(x) for clf in self.classifiers]
        return scores

        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        return np.argmax(scores)
        
        # raise NotImplementedError("Vous devez implémenter cette fonction !")