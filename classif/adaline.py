import numpy as np

from base import Classifier

class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max

        # Initialisation de w aléatoire
        self.w = np.random.randn(input_dimension) * learning_rate
        
        self.allw = [] # stockage des premiers poids
        # raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        # BOUCLE
        self.desc_set = desc_set
        self.label_set = label_set
        
        if self.history : 
            self.allw.append(self.w.copy())
        
        lastw = self.w + 1

        for i in range(self.niter_max) : 
            index = int(np.random.rand() * len(desc_set))
            x,y = desc_set[index], label_set[index]

            yhat = self.score(x)

            self.w -= self.learning_rate * x * (self.score(x) - y)

            if self.history : 
                self.allw.append(self.w.copy())

            if i%len(desc_set) == 0 : 
                if np.max(np.abs(lastw - self.w)) < 1e-3 : 
                    print("cvg in"+str(i)+"iterations")
                    return
                lastw = self.w

        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))

        # raise NotImplementedError("Please Implement this method")




# code de la classe ADALINE Analytique

class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)

        # Initialisation de w aléatoire
        # self.w = np.random.randn(input_dimension) * learning_rate
        
        # self.allw = [] # stockage des premiers poids
        # raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.w = np.linalg.solve(desc_set.T @ desc_set, desc_set.T @ label_set)

        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))

        # raise NotImplementedError("Please Implement this method")