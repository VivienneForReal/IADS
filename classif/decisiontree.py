import numpy as np
import math
# import graphviz as gv       # Import only if you want to visualize the tree
# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz

from base import Classifier

def major_class(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    ########################## COMPLETER ICI 
    valeurs, nb_fois = np.unique(Y, return_counts=True)

    return valeurs[np.argmax(nb_fois)]
    
    ##########################
        

def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropy de Shannon correspondante
    """
    ########################## COMPLETER ICI 
    s = 0.
    for i in range(len(P)) : 
        if P[i] != 0 and len(P) > 1 : 
            s -= P[i] * math.log(P[i]) / math.log(len(P))
    
    return s
    ##########################

def entropy(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropy de l'ensemble Y
    """
    ########################## COMPLETER ICI 
    _, nb_fois = np.unique(Y, return_counts=True)
    
    l = [i / len(Y) for i in nb_fois]
    return shannon(l)
    ##########################



class CategoricalNode:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def is_leaf(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def add_sons(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (CategoricalNode) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : CategoricalNode}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def add_leaf(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classify(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple 
            on rend la valeur None si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.is_leaf():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classify(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return None
    
    def count_leaves(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        nb_feuilles = 0
        if not self.Les_fils : 
            return nb_feuilles
        for fils in self.Les_fils : 
            if not self.Les_fils[fils].Les_fils : 
                nb_feuilles += 1
            else : 
                nb_feuilles += self.Les_fils[fils].count_leaves()
        return nb_feuilles
        # raise NotImplementedError("A implémenter plus tard (voir plus loin)")
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.is_leaf():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g

def construct_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropy pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    entropy_ens = entropy(Y)                        # 1
    if (entropy_ens <= epsilon):                     # 2
        # ARRET : on crée une feuille
        noeud = CategoricalNode(-1,"Label")
        noeud.add_leaf(major_class(Y))
    else:                                             # 3
        min_entropy = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropy
        # min_entropy : la valeur de l'entropy minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropy de la classe pour chaque attribut.
        for j in range(len(X[0])) : 
            entropy_attribut = 0.  # Initialisation de l'entropy de l'attribut
            valeurs_attribut = np.unique(X[:, j])  # Obtenir les valeurs uniques pour l'attribut j
            for v in valeurs_attribut:
                indices = np.where(X[:, j] == v)  # Indices où l'attribut j a la valeur v
                Y_subset = Y[indices]  # Sous-ensemble des étiquettes correspondant à ces indices
                poids = len(Y_subset) / len(Y)  # Calcul du poids de cette valeur de l'attribut
                entropy_attribut += poids * entropy(Y_subset)  # Calcul de l'entropy conditionnelle
            #print(entropy_attribut, nom_dataset.columns[i_best])
            # Mise à jour de l'attribut qui minimise l'entropy
            if entropy_attribut < min_entropy:
                min_entropy = entropy_attribut
                i_best = j
                Xbest_valeurs = valeurs_attribut

        #############################################

        if (entropy_ens - min_entropy) == 0: # pas de gain d'information possible
            # ARRET : on crée une feuille
            noeud = CategoricalNode(-1,"Label")
            noeud.add_leaf(major_class(Y))
            
        if len(LNoms)>0:  # si on a des noms de features
            noeud = CategoricalNode(i_best,LNoms[i_best])    
        else:
            noeud = CategoricalNode(i_best)
        for v in Xbest_valeurs:
            noeud.add_sons(v,construct_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
    

class DecisionTree(Classifier):
    """ Classe pour représenter un classifyur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifyur avec ses paramètres
        """
        return 'DecisionTree ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ##################
        ## COMPLETER ICI !
        ##################
        noms = [nom for nom in self.LNoms if nom != 'class']
        self.racine = construct_AD(desc_set, label_set, self.epsilon,noms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        ## COMPLETER ICI !
        ##################
        return self.racine.classify(x)
        

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.count_leaves()
    
    def draw(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropy), (liste_coupures,liste_entropys))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropy (float): entropy du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropys (List[float]): la liste des entropys correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropy = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropys trouvées et les points de coupures:
    liste_entropys = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropy de la coupure
        val_entropy_inf = entropy(cl_inf) # entropy de l'ensemble des inf
        val_entropy_sup = entropy(cl_sup) # entropy de l'ensemble des sup
        
        val_entropy = (nb_inf / float(nb_exemples)) * val_entropy_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropy_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropys trouvées:
        liste_coupures.append(v)
        liste_entropys.append(val_entropy)
        
        # si cette coupure minimise l'entropy, on mémorise ce seuil et son entropy:
        if (best_entropy > val_entropy):
            best_entropy = val_entropy
            best_seuil = v
    
    return (best_seuil, best_entropy), (liste_coupures,liste_entropys)


def partition(m_desc,m_class,n,s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    return ((m_desc[m_desc[:,n]<=s], m_class[m_desc[:,n]<=s]), \
            (m_desc[m_desc[:,n]>s], m_class[m_desc[:,n]>s]))


class NumericalNode:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def is_leaf(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def add_sons(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def add_leaf(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classify(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        #############
        # COMPLETER CETTE PARTIE 
        #
        #############
        if self.is_leaf() : 
            return self.classe
        if exemple[self.attribut] <= self.seuil : 
            return self.Les_fils["inf"].classify(exemple)
        else : 
            return self.Les_fils["sup"].classify(exemple)
        
        # raise NotImplementedError("A implémenter plus tard (voir plus loin)")

    
    def count_leaves(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        #############
        # COMPLETER CETTE PARTIE AUSSI
        #
        #############
        nb_feuilles = 0
        if not self.Les_fils : 
            return nb_feuilles
        for fils in self.Les_fils : 
            if not self.Les_fils[fils].Les_fils : 
                nb_feuilles += 1
            else : 
                nb_feuilles += self.Les_fils[fils].count_leaves()
        return nb_feuilles
        
        # raise NotImplementedError("A implémenter plus tard (voir plus loin)")
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.is_leaf():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g



def construct_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropy pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape

    _, nb_fois = np.unique(Y, return_counts=True)
    
    l = [i / len(Y) for i in nb_fois]
    
    entropy_classe = shannon(l)    # entropy(Y), on effectue des instructions facultatives vu que entropy(Y) retourne un problème incompréhensible de définition de fonction 
    
    if (entropy_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NumericalNode(-1,"Label")
        noeud.add_leaf(major_class(Y))
    else:
        gain_max = 0.0  # meilleur gain trouvé (initalisé à 0.0 => aucun gain)
        i_best = -1     # numéro du meilleur attribut (init à -1 (aucun))
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionment associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionment possible), dans ce cas, on considèrera que le
        # résultat d'un partitionment est alors ((X,Y),(None,None))       CHÚ Ý !! CHIA PHẦN
        for i in range(len(X[0])) : 
            (seuil_trouve, entropy), liste_vals = discretise(X, Y, i)
            
            if seuil_trouve is not None:  # Si on a trouvé un seuil
                gain_info = entropy_classe - entropy  # Calcul du gain d'information
                
                # Vérifier si ce gain est le meilleur trouvé jusqu'à présent
                if gain_info > gain_max:
                    gain_max = gain_info
                    i_best = i
                    Xbest_tuple = partition(X, Y, i_best, seuil_trouve)
                    Xbest_seuil = seuil_trouve
            

        
        
        ############
        if (i_best != -1): # Un attribut qui amène un gain d'information >0 a été trouvé
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NumericalNode(i_best,LNoms[i_best]) 
            else:
                noeud = NumericalNode(i_best)
                
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.add_sons( Xbest_seuil, \
                              construct_AD_num(left_data,left_class, epsilon, LNoms), \
                              construct_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NumericalNode(-1,"Label")
            noeud.add_leaf(major_class(Y))
        
    return noeud

class NumericalTree(Classifier):
    """ Classe pour représenter un classifyur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifyur avec ses paramètres
        """
        return 'DecisionTree ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construct_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classify(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.count_leaves()
    
    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------