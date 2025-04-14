import numpy as np

from classif.perceptron import Perceptron


# CLasse (abstraite) pour représenter des noyaux
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")



class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def __init__(self):
        """ Constructeur de KernelBias
            pas d'argument, les dimensions sont figées
        """
        # Appel du constructeur de la classe mère
        super().__init__(2,3)
        
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3            
            rajoute une 3e dimension au vecteur donné
        """
        
        if (V.ndim == 1): # on regarde si c'est un vecteur ou une matrice
            W = np.array([V]) # conversion en matrice
            V_proj = np.append(W,np.ones((len(W),1)),axis=1)
            V_proj = V_proj[0]  # on rend quelque chose de la même dimension
        else:
            V_proj = np.append(V,np.ones((len(V),1)),axis=1)
            
        return V_proj
        



# ------------------------ A COMPLETER :

class KernelPoly(Kernel):
    def __init__(self):
        """ Constructeur de KernelPoly
            pas d'argument, les dimensions sont figées
        """
        # Appel du constructeur de la classe mère
        super().__init__(2,6)
        
    def transform(self,V):
        """ ndarray de dim 2 -> ndarray de dim 6            
            transforme un vecteur 2D en un vecteur 6D de la forme (1, x1, x2, x1*x1, x2*x2, x1*x2)
        """
        ## TODO
        if (V.ndim == 1) : 
          W = np.array([V]) # conversion en matrice
          V_proj = np.append(np.ones((len(W),1)),W,axis=1)
          x1, x2 = V[0], V[1]

          
          V_proj = V_proj[0]  # on rend quelque chose de la même dimension
          V_proj = np.append(V_proj, x1*x1)
          V_proj = np.append(V_proj, x2*x2)
          V_proj = np.append(V_proj, x1*x2)
        
        else : 
          V_proj = np.append(np.ones((len(V),1)),V,axis=1)
          x1, x2 = V[:,0], V[:,1]
          
          one = np.ones((len(V),1))
          x11 = x1 * x1
          x22 = x2 * x2
          x12 = x1 * x2

          V_proj = np.append(V_proj, [one[i] * x11[i] for i in range(len(one))],axis=1)
          V_proj = np.append(V_proj, [one[i] * x22[i] for i in range(len(one))],axis=1)
          V_proj = np.append(V_proj, [one[i] * x12[i] for i in range(len(one))],axis=1)

        return V_proj
        # raise NotImplementedError("Please Implement this method")





# ------------------------ A COMPLETER :
class KernelPerceptron(Perceptron):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        super().__init__(input_dimension, learning_rate, bool(init))
        self.kernel = noyau

        # --- Initialisation de w ---
        self.w = np.zeros(self.kernel.get_output_dim())
        if not init :
            for i in range(self.kernel.get_output_dim()):
                self.w[i] = ((2*np.random.random()-1) * 0.001)

        # self.allw =[self.w.copy()] # stockage des premiers poids
        
        # raise NotImplementedError("Please Implement this method")
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        # Mélanger les indices des exemples
        data = np.array([(i,desc_set[i],label_set[i]) for i in range(len(desc_set))],dtype=object)
        np.random.shuffle(data)

        for _,x_i,y_i in data :
            predict = self.predict(x_i)

            if y_i != predict : 
                self.w += self.learning_rate * y_i * self.kernel.transform(x_i)

            #self.allw.append(self.w.copy())
        
        return self.w 

        #raise NotImplementedError("Please Implement this method")
     
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        # Projeter la description dans l'espace de dimension supérieure
        x_transformed = self.kernel.transform(x)
        # Calculer le score de prédiction
        score = np.dot(self.w, x_transformed)
        # Appliquer la fonction d'activation
        return score
        #raise NotImplementedError("Please Implement this method")
    