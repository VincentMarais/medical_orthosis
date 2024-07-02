#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import btk #conda install -c conda-forge btk
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



# from mp1_toolkits.mplot3d import Axes3D

## Dataset d'une femme de 74.5kg atteint de sclérose en plaque

def read_c3d(filename):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()
    return acq

file = "c:\\Users\\vince\\Desktop\\LAM_Cath_Session_01-btk_Moyenne.c3d"
acq = read_c3d(file)
file = "c:\\Users\\vince\Desktop\\LAM Cath Session 01 - btk_Ecart-type.c3d"
acq1 = read_c3d(file)
## Angle du genou

LeftKneeAnglesMean = acq.GetPoint("LKneeAngles").GetValues()
LeftKneeAnglesStd = acq1.GetPoint("LKneeAngles").GetValues()
RightKneeAnglesMean = acq.GetPoint("RKneeAngles").GetValues()
RightKneeAnglesStd = acq1.GetPoint("RKneeAngles").GetValues()
plt.plot(range(101),LeftKneeAnglesMean[:,0],'r', label='Angle moyen gauche')
plt.plot(range(101),LeftKneeAnglesMean[:,0]+LeftKneeAnglesStd[:,0],'r--')
plt.plot(range(101),LeftKneeAnglesMean[:,0]-LeftKneeAnglesStd[:,0],'r--')
plt.fill_between(range(101),LeftKneeAnglesMean[:,0]-LeftKneeAnglesStd[:,0],LeftKneeAnglesMean[:,0]+LeftKneeAnglesStd[:,0],alpha=0.7,color='salmon')
plt.plot(range(101),RightKneeAnglesMean[:,0],'b', label='Angle moyen droit')
plt.plot(range(101),RightKneeAnglesMean[:,0]+RightKneeAnglesStd[:,0],'b--')
plt.plot(range(101),RightKneeAnglesMean[:,0]-RightKneeAnglesStd[:,0],'b--')
plt.fill_between(range(101),RightKneeAnglesMean[:,0]-RightKneeAnglesStd[:,0],RightKneeAnglesMean[:,0]+RightKneeAnglesStd[:,0],alpha=0.7,color='skyblue')
plt.title('Angle F/E Genou')
plt.xlabel('Cycle de marche (%)')
plt.ylabel('Angle du genou (°)')
plt.legend()
plt.show()

## Moment du genoux
LeftKneeMomentsMean = acq.GetPoint("LKneeMoment").GetValues()
LeftKneeMomentsStd = acq1.GetPoint("LKneeMoment").GetValues()
RightKneeMomentsMean = acq.GetPoint("RKneeMoment").GetValues()
RightKneeMomentsStd = acq1.GetPoint("RKneeMoment").GetValues()
plt.plot(range(101),LeftKneeMomentsMean[:,0]*74.5/1000,'r', label='Couple Moyen gauche')
plt.plot(range(101),(LeftKneeMomentsMean[:,0]+LeftKneeMomentsStd[:,0])*74.5/1000,'r--')
plt.plot(range(101),(LeftKneeMomentsMean[:,0]-LeftKneeMomentsStd[:,0])*74.5/1000,'r--')
plt.fill_between(range(101),(LeftKneeMomentsMean[:,0]-LeftKneeMomentsStd[:,0])*74.5/1000,(LeftKneeMomentsMean[:,0]+LeftKneeMomentsStd[:,0])*74.5/1000,alpha=0.7,color='salmon')
plt.plot(range(101),RightKneeMomentsMean[:,0]*74.5/1000,'b', label='Couple Moyen droit')
plt.plot(range(101),(RightKneeMomentsMean[:,0]+RightKneeMomentsStd[:,0])*74.5/1000,'b--')
plt.plot(range(101),(RightKneeMomentsMean[:,0]-RightKneeMomentsStd[:,0])*74.5/1000,'b--')
plt.fill_between(range(101),(RightKneeMomentsMean[:,0]-RightKneeMomentsStd[:,0])*74.5/1000,(RightKneeMomentsMean[:,0]+RightKneeMomentsStd[:,0])*74.5/1000,alpha=0.7,color='skyblue')
plt.title('Moment F/E Genou') # Moment en N.m mais initialement c'était en N.mm/kg
plt.xlabel('Cycle de marche (%)')
plt.ylabel('Moment du genou (N.m)')
plt.legend()
plt.show()

## EMG Biceps fémoral
LeftIJMean = acq.GetAnalog('Biceps Femoris Gauche').GetValues() + 2*acq.GetAnalog('Biceps Femoris Gauche').GetOffset()
LeftIJStd = acq1.GetAnalog('Biceps Femoris Gauche').GetValues() + 2*acq1.GetAnalog('Biceps Femoris Gauche').GetOffset()
RightIJMean = acq.GetAnalog('Biceps Femoris Droit').GetValues() + 2*acq.GetAnalog('Biceps Femoris Droit').GetOffset()
RightIJStd = acq1.GetAnalog('Biceps Femoris Droit').GetValues() + 2*acq1.GetAnalog('Biceps Femoris Droit').GetOffset()
plt.plot(range(101),LeftIJMean[:,0],'r', label='EMG Moyen gauche')
plt.plot(range(101),LeftIJMean[:,0]+LeftIJStd[:,0],'r--')
plt.plot(range(101),LeftIJMean[:,0]-LeftIJStd[:,0],'r--')
plt.fill_between(range(101),LeftIJMean[:,0]-LeftIJStd[:,0],LeftIJMean[:,0]+LeftIJStd[:,0],alpha=0.7,color='salmon')
plt.plot(range(101),RightIJMean[:,0],'b', label='EMG Moyen droit')
plt.plot(range(101),RightIJMean[:,0]+RightIJStd[:,0],'b--')
plt.plot(range(101),RightIJMean[:,0]-RightIJStd[:,0],'b--')
plt.fill_between(range(101),RightIJMean[:,0]-RightIJStd[:,0],RightIJMean[:,0]+RightIJStd[:,0],alpha=0.7,color='skyblue')
plt.title('EMG Biceps Fémoral')
plt.xlabel('Cycle de marche (%)')
plt.ylabel('EMG Biceps fémoral (Volt)')
plt.legend()
plt.show()


## EMG: Vaste latéral
LeftVLMean = acq.GetAnalog('Vastus Lateralis Gauche').GetValues() + 2*acq.GetAnalog('Vastus Lateralis Gauche').GetOffset()
LeftVLStd = acq1.GetAnalog('Vastus Lateralis Gauche').GetValues() + 2*acq1.GetAnalog('Vastus Lateralis Gauche').GetOffset()
RightVLMean = acq.GetAnalog('Vastus Lateralis Droit').GetValues() + 2*acq.GetAnalog('Vastus Lateralis Droit').GetOffset()
RightVLStd = acq1.GetAnalog('Vastus Lateralis Droit').GetValues() + 2*acq1.GetAnalog('Vastus Lateralis Droit').GetOffset()
plt.plot(range(101),LeftVLMean[:,0],'r', label='EMG Moyen gauche')
plt.plot(range(101),LeftVLMean[:,0]+LeftVLStd[:,0],'r--')
plt.plot(range(101),LeftVLMean[:,0]-LeftVLStd[:,0],'r--')
plt.fill_between(range(101),LeftVLMean[:,0]-LeftVLStd[:,0],LeftVLMean[:,0]+LeftVLStd[:,0],alpha=0.7,color='salmon')
plt.plot(range(101),RightVLMean[:,0],'b', label='EMG Moyen droit')
plt.plot(range(101),RightVLMean[:,0]+RightVLStd[:,0],'b--')
plt.plot(range(101),RightVLMean[:,0]-RightVLStd[:,0],'b--')
plt.fill_between(range(101),RightVLMean[:,0]-RightVLStd[:,0],RightVLMean[:,0]+RightVLStd[:,0],alpha=0.7,color='skyblue')
plt.title('EMG Vaste latéral')
plt.xlabel('Cycle de marche (%)')
plt.ylabel('EMG Vaste latéral (Volt)')
plt.legend()
plt.show()

## EMG: Droit fémoral
LeftRFMean = acq.GetAnalog('Rectus Femoris Gauche').GetValues() + 2*acq.GetAnalog('Rectus Femoris Gauche').GetOffset()
LeftRFStd = acq1.GetAnalog('Rectus Femoris Gauche').GetValues() + 2*acq1.GetAnalog('Rectus Femoris Gauche').GetOffset()
RightRFMean = acq.GetAnalog('Rectus Femoris Droit').GetValues() + 2*acq.GetAnalog('Rectus Femoris Droit').GetOffset()
RightRFStd = acq1.GetAnalog('Rectus Femoris Droit').GetValues() + 2*acq1.GetAnalog('Rectus Femoris Droit').GetOffset()
plt.plot(range(101),LeftRFMean[:,0],'r', label='EMG Moyen gauche' )
plt.plot(range(101),LeftRFMean[:,0]+LeftRFStd[:,0],'r--')
plt.plot(range(101),LeftRFMean[:,0]-LeftRFStd[:,0],'r--')
plt.fill_between(range(101),LeftRFMean[:,0]-LeftRFStd[:,0],LeftRFMean[:,0]+LeftRFStd[:,0],alpha=0.7,color='salmon')
plt.plot(range(101),RightRFMean[:,0],'b', label='EMG Moyen droit')
plt.plot(range(101),RightRFMean[:,0]+RightRFStd[:,0],'b--')
plt.plot(range(101),RightRFMean[:,0]-RightRFStd[:,0],'b--')
plt.fill_between(range(101),RightRFMean[:,0]-RightRFStd[:,0],RightRFMean[:,0]+RightRFStd[:,0],alpha=0.7,color='skyblue')
plt.xlabel('Cycle de marche (%)')
plt.ylabel('EMG Droit fémoral (Volt)')
plt.legend()
plt.title('EMG Droit fémoral')
plt.show()

## Pré-analyse dataset

Biceps_Femoral= RightIJMean

Droit_Femoral=RightRFMean

Vaste_lateral=RightVLMean

Angle_droit=np.delete(RightKneeAnglesMean,[1,2],1)

Moments_droit=np.delete(RightKneeMomentsMean,[1,2],1) # On supprimer les colonnes [1,2] , 1: colonne et si on met 0 on supprime les lignes 1 et 2

Moments_droit=np.multiply(Moments_droit, 10**-3) ## On normalise Nm/kg les données


## Affichage de la base de donnée Moment
plt.scatter(Droit_Femoral,Moments_droit)
plt.xlabel('EMG (Volt)')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.plot()
plt.title('Dataset EMG Droit Fémoral et Moment droit du genou normalisé')
plt.show()

plt.scatter(Biceps_Femoral,Moments_droit)
plt.xlabel('EMG (Volt) ')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.plot()
plt.title('Dataset EMG Biceps Fémoral et Moment droit du genou normalisé')
plt.show()

plt.scatter(Vaste_lateral,Moments_droit)
plt.xlabel('EMG (Volt)')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.plot()
plt.title('Dataset EMG Vaste latéral et Moment droit du genou normalisé')
plt.show()

## Affichage de la base de donnée Angles

plt.scatter(Droit_Femoral,Angle_droit)
plt.xlabel('EMG ')
plt.ylabel('Angle du genou')
plt.plot()
plt.show()

plt.scatter(Biceps_Femoral,Angle_droit)
plt.xlabel('EMG ')
plt.ylabel('Angle du genou')
plt.plot()
plt.show()

plt.scatter(Vaste_lateral,Angle_droit)
plt.xlabel('EMG ')
plt.ylabel('Angle du genou')
plt.plot()
plt.show()



## On crée la matrice X avec le modéle tau = u^a*exp(b-u*c)

X_1=np.hstack(((np.log(Biceps_Femoral), np.ones(Biceps_Femoral.shape), -Biceps_Femoral ))) # ln(u) / 1 / -u

print(X_1)
X_2=np.hstack(((np.log(Droit_Femoral), np.ones(Droit_Femoral.shape), -Droit_Femoral ))) # ln(u) / 1 / -u

print(X_2)

X_3=np.hstack(((np.log(Vaste_lateral), np.ones(Vaste_lateral.shape), -Vaste_lateral ))) # ln(u) / 1 / -u

print(X_3)

print(X_3.T)

## On crée la matrice X avec le modéle tau = a + bu^2 + c*exp(u)

X_12=np.hstack((( np.ones(Biceps_Femoral.shape), Biceps_Femoral**2, np.exp(Biceps_Femoral) )))


X_22=np.hstack((( np.ones(Droit_Femoral.shape), Droit_Femoral**2, np.exp(Droit_Femoral) )))


X_32=np.hstack((( np.ones(Vaste_lateral.shape), Vaste_lateral**2, np.exp(Vaste_lateral) )))

##Modèle polynomial

X_13=np.hstack((( Biceps_Femoral**4  , Biceps_Femoral**3, Biceps_Femoral**2, Biceps_Femoral, np.ones(Biceps_Femoral.shape) )))


X_23=np.hstack((( Droit_Femoral**4  , Droit_Femoral**3, Droit_Femoral**2, Droit_Femoral, np.ones(Droit_Femoral.shape) )))

X_33=np.hstack((( Vaste_lateral**4  , Vaste_lateral**3, Vaste_lateral**2, Vaste_lateral, np.ones(Vaste_lateral.shape) )))

## dEGRE 6
X_13=np.hstack((( Biceps_Femoral**6  , Biceps_Femoral**3, Biceps_Femoral**2, Biceps_Femoral, np.ones(Biceps_Femoral.shape) )))


X_23=np.hstack((( Droit_Femoral**6, Droit_Femoral**5, Droit_Femoral**4, Droit_Femoral**3, Droit_Femoral**2, Droit_Femoral, np.ones(Droit_Femoral.shape) )))

X_33=np.hstack((( Vaste_lateral**4  , Vaste_lateral**3, Vaste_lateral**2, Vaste_lateral, np.ones(Vaste_lateral.shape) )))


## Degre 8
X_23=np.hstack((( Droit_Femoral**8, Droit_Femoral**7, Droit_Femoral**6, Droit_Femoral**5, Droit_Femoral**4, Droit_Femoral**3, Droit_Femoral**2, Droit_Femoral, np.ones(Droit_Femoral.shape) )))

## ## On créé le vecteur theta
theta = np.random.randn(3,1) # [a,b,c]
print(theta)
##  Modèle

def F_expo(X,theta): # F = exp(X * \theta) ou tau = u^a*exp(b-u*c)
    return np.exp(X.dot(theta))

def modele(X,theta): # tau = a + bu^2 + c*exp(u) + Modele polynomial
    return X.dot(theta)

## Fonction MSE

def MSE_1(X,y,theta): # tau = u^a*exp(b-u*c)
    m= len(y)
    return 1/(2*m)*np.sum((F_expo(X,theta) - y)**2)

def MSE_2(X,y,theta): # tau = a + bu^2 + c*exp(u) + Modele polynomial
    m= len(y)
    return 1/(2*m)*np.sum((modele(X,theta) - y)**2)

## Gradient

def grad_1(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(F_expo(X, theta) - y)

def grad_2(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(modele(X, theta) - y)

## Descente du gradient  1: u^a*exp(b-u*c)
def gradient_descent_1(X, y, theta, learning_rate, n_iterations):
    # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    Cout_temporel = np.zeros(n_iterations)

    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad_1(X, y, theta) # mise a jour du parametre theta (formule du gradient descent)
        Cout_temporel[i] = MSE_1(X, y, theta) # on enregistre la valeur du Cout au tour i dans Erreur_tempo[i]
    return theta, Cout_temporel

## Descente du gradient 2: a + bu^2 + c*exp(u) +  Modèle polynomial

def gradient_descent_2(X, y, theta, learning_rate, n_iterations):
    # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    Erreur_tempo = np.zeros(n_iterations)

    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad_2(X, y, theta) # mise a jour du parametre theta (formule du gradient descent)
        Erreur_tempo[i] = MSE_2(X, y, theta) # on enregistre la valeur du Cout au tour i dans Erreur_tempo[i]

    return theta, Erreur_tempo
## R2
def rcarre(y, pred):
    u = ((y - pred)**2).sum() #residu de la somme des carres
    v = ((y - y.mean())**2).sum()# somme totale des carres
    return 1 - u/v

## Indice mini d'une liste
def Indice_min_liste(L):
    N=len(L)
    min=L[0]
    indice_min=0
    for i in range (0,N):
        if min>L[i]:
            min=L[i]
            indice_min=i
    return indice_min

## Meilleur learning_rate
def meilleur_learning_rate(x,y,learning_rate_max, n_iteration):
    i=0.001
    L_ecart=[]
    L_learning=[]
    X=np.hstack(((np.log(x), np.ones(x.shape), -x ))) # ln(u) / 1 / -u
    theta=np.random.randn(3,1)

    while i < learning_rate_max:
        theta_final= gradient_descent_1(X, y, theta, learning_rate, n_iterations)[0]
        predictions = F_expo(X, theta_final)
        if rcarre(y, predictions) > 0:
            L_ecart.append([1-rcarre(y, predictions)])
            L_learning.append(i)
        i+=0.001
        #Indice = Indice_min_liste(L_ecart)
    return  L_learning[Indice] # meilleur_learning_rate(x=Droit_Femoral,y=Moments_droit,learning_rate_max = 1, n_iteration =3000)


## Analyse final modele a + bu^2 + c*exp(u) (Descente du gradient)

n_iterations = 1000
learning_rate = 0.001

theta_final, Erreur_tempo = gradient_descent_1(X_22, Moments_droit, theta, learning_rate, n_iterations)

print(theta_final) # voici les parametres du modele une fois que la machine a été entrainée

# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = modele(X_22, theta_final)

# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(Droit_Femoral, Moments_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.show()

## Analyse final modele tau = u^a*exp(b-u*c) (Descente du gradient)

n_iterations = 3000
learning_rate = 0.001

theta_final, Erreur_tempo = gradient_descent_1(X_2, Moments_droit, theta, learning_rate, n_iterations)

print(theta_final) # voici les parametres du modele une fois que la machine a été entrainée

# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = F_expo(X_2, theta_final)

# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(Droit_Femoral, Moments_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.xlabel('EMG (Volt)')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.title('Dataset EMG Droit Fémoral et Moment droit du genou normalisé')

plt.show()

print('R2=', rcarre(Moments_droit, predictions))

plt.plot(range(3000), Erreur_tempo) # Evolution de l'erreur
plt.show()

## Analyse final modele polynomial (Descente du gradient)
theta = np.random.randn(9,1) # [a,b,c]
# X_p = creation_polynome(Droit_Femoral,4)
n_iterations = 10000
learning_rate = 1.5 # Avec 1.5 on a R2= 0.73695233867375076

X_p=np.hstack((( Droit_Femoral**8, Droit_Femoral**7, Droit_Femoral**6, Droit_Femoral**5, Droit_Femoral**4, Droit_Femoral**3, Droit_Femoral**2, Droit_Femoral, np.ones(Droit_Femoral.shape) )))


#X_p=np.hstack((( Droit_Femoral**2, Droit_Femoral, np.ones(Droit_Femoral.shape) )))

theta_final, Erreur_tempo = gradient_descent_2(X_p, Moments_droit, theta, learning_rate, n_iterations)

print(theta_final) # voici les parametres du modele une fois que la machine a été entrainée

# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = modele(X_p, theta_final)

# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(Droit_Femoral, Moments_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.xlabel('EMG (Volt)')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.title('Dataset EMG Droit Fémoral et Moment droit du genou normalisé')
plt.show()
print('R2=', rcarre(Moments_droit, predictions))



## Déterminer le degre du modele polynomial le plus opti (Descente de gradient)
def polynome_opti_grad(n,x,y,learning_rate,n_iterations): # n degre max du polynôme
    X=np.array(np.ones(x.shape))
    L_ecart=[]
    L_degre=[]
    L_r2=[]
    theta=np.random.randn(2,1)

    for i in range(1,n+1):
        X=np.hstack((( x**i, X )))

        theta_final=gradient_descent_2(X, y, theta, learning_rate, n_iterations)[0]
        predictions = modele(X, theta_final)
        if rcarre(y, predictions) > 0:
            L_ecart.append([1-rcarre(y, predictions)])
            L_degre.append(i)
            L_r2.append(rcarre(y, predictions))

    min_ecart=Indice_min_liste(L_ecart)
    degre_poly_opti= L_degre[min_ecart]
    P = creation_polynome(x, degre_poly_opti)
    return degre_poly_opti , P, L_ecart, L_degre, L_r2 # polynome_opti_grad(10,Droit_Femoral,Moments_droit, 1.5, 10000)

## Polynome creation
def creation_polynome(x,n):
    L=[]
    X=np.array(np.ones(x.shape))
    for i in range (1,n+1):
        X=np.hstack((( x**i, X )))
        L.append(i)
    return X

##Equation normale final Droit_Femoral (Sans polynome opti)
X_p=np.hstack((( Droit_Femoral**8, Droit_Femoral**7, Droit_Femoral**6, Droit_Femoral**5, Droit_Femoral**4, Droit_Femoral**3, Droit_Femoral**2, Droit_Femoral, np.ones(Droit_Femoral.shape) )))


#X_p=np.hstack((( Droit_Femoral**2, Droit_Femoral, np.ones(Droit_Femoral.shape) )))

theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Moments_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Droit_Femoral, Moments_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.xlabel('EMG (Volt)')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.title('Dataset EMG Droit Fémoral et Moment droit du genou normalisé')
plt.show()

print('R2=', rcarre(Moments_droit, predictions))

## Déterminer le degre du modele polynomial le plus opti (Eq Normale)
def polynome_opti_normal(n,x,y): # n degre max du polynôme
    X=np.array(np.ones(x.shape))
    L_ecart=[]
    L_degre=[]
    L_r2=[]

    for i in range(1,n+1):
        X=np.hstack((( x**i, X )))

        theta_final=np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        predictions = modele(X, theta_final)
        if rcarre(y, predictions) > 0:
            L_ecart.append([1-rcarre(y, predictions)])
            L_degre.append(i)
            L_r2.append(rcarre(y, predictions))

    min_ecart=Indice_min_liste(L_ecart)
    degre_poly_opti= L_degre[min_ecart]
    P = creation_polynome(x, degre_poly_opti)
    return degre_poly_opti , P, L_ecart, L_degre, L_r2 # polynome_opti_normal(10,Droit_Femoral,Moments_droit)





## Famille de log

X_p=polynome_opti_normal_log(101,Droit_Femoral,Moments_droit)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Moments_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Droit_Femoral, Moments_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.xlabel('EMG (Volt)')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.title('Dataset EMG Droit Fémoral et Moment droit du genou normalisé')
plt.show()

print('R2=', rcarre(Moments_droit, predictions))
## Equation normale final Vaste_lateral

X_p=polynome_opti_normal(60,Vaste_lateral,Moments_droit)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Moments_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Vaste_lateral, Moments_droit)
plt.plot(Vaste_lateral, predictions, c='r')
plt.show()

print('R2=', rcarre(Moments_droit, predictions))

## Equation normale final Biceps_Femoral

X_p=polynome_opti_normal(60,Biceps_Femoral,Moments_droit)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Moments_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Biceps_Femoral, Moments_droit)
plt.plot(Biceps_Femoral, predictions, c='r')
plt.show()

print('R2=', rcarre(Moments_droit, predictions))


## Equation normale final Droit_Femoral (Avec polynome opti)

X_p=polynome_opti_normal(101,Droit_Femoral,Moments_droit)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Moments_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Droit_Femoral, Moments_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.xlabel('EMG (Volt)')
plt.ylabel('Moment du genou normalisé (N.m/kg)')
plt.title('Dataset EMG Droit Fémoral et Moment droit du genou normalisé')
plt.show()

print('R2=', rcarre(Moments_droit, predictions))

## Représentation final Moment predit et moment réel
plt.plot(range(101), predictions, 'r', label='Prédiction du moment du genou droit' )
plt.plot(range(101),RightKneeMomentsMean[:,0]*10**-3,'b', label='Moment moyen du genou droit')
plt.title('Moment du Genou droit')
plt.xlabel('Cycle de marche (%)')
plt.ylabel('Moment du Genou droit Normalisé (N.m/kg)')
plt.legend()
plt.show()



#### ANGLES

## Equation normale final Droit_Femoral

X_p=polynome_opti_normal(60,Droit_Femoral,Angle_droit)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Angle_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Droit_Femoral, Angle_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.show()

print('R2=', rcarre(Angle_droit, predictions))




## Equation normale final Vaste_lateral

X_p=polynome_opti_normal(60,Vaste_lateral,Angle_droit)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Angle_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Vaste_lateral, Angle_droit)
plt.plot(Vaste_lateral, predictions, c='r')
plt.show()

print('R2=', rcarre(Angle_droit, predictions))

## Equation normale final Biceps_Femoral

X_p=polynome_opti_normal(60,Biceps_Femoral,Angle_droit)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Angle_droit)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Biceps_Femoral, Angle_droit)
plt.plot(Biceps_Femoral, predictions, c='r')
plt.show()

print('R2=', rcarre(Angle_droit, predictions))

## Représentation final Angle predit et Angle réel
plt.plot(range(101), predictions, 'r', label='Angle prédit )
plt.plot(range(101),RightKneeAnglesMean[:,0],'b', label='Angle genou droit')
plt.title('Angle du Genou droit')
plt.xlabel('Cycle de marche (%)')
plt.ylabel('Angle du Genou droit Normalisé (degre)')
plt.legend()
plt.show()


## Juste pour la flexion

Flexion_Moment=Moments_droit[27:40] # On sélectionne les éléments de la ligne de 27 à 40

Flexion_Droit_Femoral=Droit_Femoral[27:40]

X_p=polynome_opti_normal(10,Flexion_Droit_Femoral,Flexion_Moment)[1]
theta_final_normale= np.linalg.pinv(X_p.T.dot(X_p)).dot(X_p.T).dot(Flexion_Moment)

predictions = modele(X_p, theta_final_normale)
plt.scatter(Flexion_Droit_Femoral, Flexion_Moment)
plt.plot(Flexion_Droit_Femoral, predictions, c='r')
plt.show()

print('R2=', rcarre(Flexion_Moment, predictions))


plt.plot(range(13), predictions, 'r', label='Prédiction du couple' )
plt.plot(range(13),Flexion_Moment,'b', label='Couple Moyen droit')
plt.title('Moment de la mise en flexion du Genou droit')
plt.legend()
plt.show()



## Avec deux features TEST VISUALISATION 3D
Muscles_2= np.hstack((( Droit_Femoral, Vaste_lateral )))
X_f2=np.hstack((( Muscles_2, np.ones(Droit_Femoral.shape) )))

theta_final_normale_2= np.linalg.pinv(X_f2.T.dot(X_f2)).dot(X_f2.T).dot(Moments_droit)
predictions = modele(X_f2, theta_final_normale_2)

fig= plt.figure()

ax = fig.add_subplot(111,projection="3d")

ax.scatter(Vaste_lateral, Moments_droit)
ax.scatter(Droit_Femoral, Moments_droit)
plt.show()


## Polynôme avec 2 features
def polynome_opti_normal_features_2(n,x_1,x_2,y): # n degre max du polynôme
    X=np.hstack((( x_1, x_2, np.ones(x.shape) )))
    L_ecart=[]
    L_degre=[]
    theta=np.random.randn(2,1)

    for i in range(2,n+1):
        theta_final=np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        predictions = modele(X, theta_final)
        if rcarre(y, predictions) > 0:
            L_ecart.append([1-rcarre(y, predictions)])
            L_degre.append(i)

        X=np.hstack((( x_1**i, x_2**i, X )))
        theta=np.random.randn(i+3,1) # ex: on a plus d'imdéterminer deg(n) ou n>2 il nous faut un theta de dimension n+2

    min_ecart=Indice_min_liste(L_ecart)
    degre_poly_opti= L_degre[min_ecart]
    return degre_poly_opti,X # polynome_opti_normal(10,Droit_Femoral,Moments_droit)

theta_final_normale= np.linalg.pinv(X_f3.T.dot(X_f3)).dot(X_f3.T).dot(Moments_droit)

## Modele avec le log
def polynome_opti_normal_log(n,x,y): # n degre max du polynôme
    X=np.array(np.ones(x.shape))
    L_ecart=[]
    L_degre=[]
    L_r2=[]
    theta=np.random.randn(2,1)

    for i in range(1,n+1):
        X=np.hstack((( np.log(x)**i, X )))

        theta_final=np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        predictions = modele(X, theta_final)
        if rcarre(y, predictions) > 0:
            L_ecart.append([1-rcarre(y, predictions)])
            L_degre.append(i)
            L_r2.append(rcarre(y, predictions))

    min_ecart=Indice_min_liste(L_ecart)
    degre_poly_opti= L_degre[min_ecart]
    P = creation_polynome(x, degre_poly_opti)
    return degre_poly_opti , P, L_ecart, L_degre, L_r2 # polynome_opti_normal_log(10,Droit_Femoral,Moments_droit)

## BIC

poly_reg = PolynomialFeatures(degree=35)
X_poly = poly_reg.fit_transform(Droit_Femoral)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, Moments_droit)


plt.scatter(Droit_Femoral, Moments_droit)
plt.plot(Droit_Femoral, pol_reg.predict(poly_reg.fit_transform(Droit_Femoral)), color='r')
plt.title('Modele')
plt.xlabel('EMG')
plt.ylabel('Moment')
plt.show()

plt.plot(range(101), pol_reg.predict(poly_reg.fit_transform(Droit_Femoral)), 'r', label='Prédiction du moment du genou droit' )
plt.plot(range(101),RightKneeMomentsMean[:,0]*10**-3,'b', label='Moment moyen du genou droit')
plt.title('Moment du Genou droit')
plt.xlabel('Cycle de marche (%)')
plt.ylabel('Moment du Genou droit Normalisé (N.m/kg)')
plt.legend()
plt.show()

def max(L):
    max=L[0]
    indice=0

    for i in range (len(L)):
        if L[i]>max:
            max=L[i]
            indice=i
    return indice



def poly_opti_ultime(n,x,y): # poly_opti_ultime(101,Droit_Femoral,Moments_droit)
    D=[]
    BIC=[]
    R2=[]

    for degree in range (1,n+1):
        poly_reg = PolynomialFeatures(degree)
        X_poly = poly_reg.fit_transform(x)
        model=sm.OLS(y,X_poly).fit()
        y_pred=model.predict(X_poly)

        D.append(degree)
        BIC.append(model.bic)
        R2.append(model.rsquared)
        i=Indice_min_liste(BIC)
        i_n=max(R2)
        D_opti=D[i]

        D_n=D[i_n]
    return D[i_n], BIC, R2







