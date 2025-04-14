import numpy as np

from base import Classifier

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, num_labels=10, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - num_labels (int): nombre de classes dans la classification
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        super().__init__(input_dimension)
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        # --- Initialisation de w ---
        self.w = np.zeros((num_labels, input_dimension))  # Each label gets its own weight vector
        if not init:
            for i in range(num_labels):
                for j in range(input_dimension):
                    self.w[i, j] = ((2*np.random.random()-1) * 0.001)

        self.allw = [self.w.copy()]  # stockage des premiers poids

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """   
        # Choisir aléatoirement un exemple x_i de X
        data = list(zip(desc_set, label_set))
        np.random.shuffle(data)

        for x_i, y_i in data:
            predict = self.predict(x_i)
            if y_i != predict:
                self.w[y_i] += self.learning_rate * x_i
                self.w[predict] -= self.learning_rate * x_i

            self.allw.append(self.w.copy())
        
        return self.w 

    def score(self, x):
        """ rend le score de prédiction sur x (un vecteur de scores pour chaque classe)
            x: une description
        """
        return np.dot(self.w, x)

    def predict(self, x):
        """ rend la prediction sur x (un chiffre de 0 à 9)
            x: une description
        """
        scores = self.score(x)
        return np.argmax(scores)

    def get_allw(self):
        return self.allw

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """  
        norm_diff_values = []      
        i = 0
        
        while i < nb_max: 
            w_old = self.w.copy()
            self.w = self.train_step(desc_set, label_set)
            
            norm_diff = np.linalg.norm(w_old - self.w)
            norm_diff_values.append(norm_diff)
            
            if norm_diff < seuil:
                break
            
            i += 1
        
        return norm_diff_values
    

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate=learning_rate, init=init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        ##################
        ### A COMPLETER !
        ##################
        # Ne pas oublier d'ajouter les poids à allw avant de terminer la méthode
        self.desc_set = desc_set
        self.label_set = label_set
        data = np.array(
            [(i,desc_set[i],label_set[i]) \
                for i in range(len(desc_set))],dtype=object
            )
        
        np.random.shuffle(data)
        for _, x_i, y_i in data : 
            f_x_i = self.score(x_i)

            if f_x_i * y_i < 1 : 
                self.w += self.learning_rate * (y_i - f_x_i) * x_i
            
            self.allw.append(self.w.copy())
        return self.w
        # raise NotImplementedError("Vous devez implémenter cette méthode !")    
# ------------------------ 