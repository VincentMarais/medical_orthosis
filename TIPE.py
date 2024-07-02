#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import btk #conda install -c conda-forge btk
import matplotlib.pyplot as plt
import numpy.linalg as alg
## Dataset d'une femme de 74.5kg atteint de sclérose en plaque
def read_c3d(filename): # Fonction pour lire les fichiers en c3d
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()
    return acq

file = "c:\\Users\\vince\\Desktop\\LAM_Cath_Session_01-btk_Moyenne.c3d"
acq = read_c3d(file)
file = "c:\\Users\\vince\Desktop\\LAM Cath Session 01 - btk_Ecart-type.c3d" #
acq1 = read_c3d(file)

## Angle du genou
LeftKneeAnglesMean = acq.GetPoint("LKneeAngles").GetValues()
LeftKneeAnglesStd = acq1.GetPoint("LKneeAngles").GetValues()
RightKneeAnglesMean = acq.GetPoint("RKneeAngles").GetValues()
RightKneeAnglesStd = acq1.GetPoint("RKneeAngles").GetValues()
plt.plot(range(101),LeftKneeAnglesMean[:,0],'r')
plt.plot(range(101),LeftKneeAnglesMean[:,0]+LeftKneeAnglesStd[:,0],'r--')
plt.plot(range(101),LeftKneeAnglesMean[:,0]-LeftKneeAnglesStd[:,0],'r--')
plt.fill_between(range(101),LeftKneeAnglesMean[:,0]-LeftKneeAnglesStd[:,0],LeftKneeAnglesMean[:,0]+LeftKneeAnglesStd[:,0],alpha=0.7,color='salmon')
plt.plot(range(101),RightKneeAnglesMean[:,0],'b')
plt.plot(range(101),RightKneeAnglesMean[:,0]+RightKneeAnglesStd[:,0],'b--')
plt.plot(range(101),RightKneeAnglesMean[:,0]-RightKneeAnglesStd[:,0],'b--')
plt.fill_between(range(101),RightKneeAnglesMean[:,0]-RightKneeAnglesStd[:,0],RightKneeAnglesMean[:,0]+RightKneeAnglesStd[:,0],alpha=0.7,color='skyblue')
plt.title('Angle F/E Genou')
plt.show()


## Moment du genoux
LeftKneeMomentsMean = acq.GetPoint("LKneeMoment").GetValues() # array (101, 3) 101 lignes et 3 colonnes (pourquoi il y a trois colonnes ?) que représente [:,1] et [:,2]
LeftKneeMomentsStd = acq1.GetPoint("LKneeMoment").GetValues()
RightKneeMomentsMean = acq.GetPoint("RKneeMoment").GetValues()
RightKneeMomentsStd = acq1.GetPoint("RKneeMoment").GetValues()

## Moment du genou gauche non normalisé
plt.plot(range(101),LeftKneeMomentsMean[:,0]*74.5/1000,'r')
plt.show()
plt.plot(range(101),(LeftKneeMomentsMean[:,0]+LeftKneeMomentsStd[:,0])*74.5/1000,'r--')
plt.plot(range(101),(LeftKneeMomentsMean[:,0]-LeftKneeMomentsStd[:,0])*74.5/1000,'r--')
plt.fill_between(range(101),(LeftKneeMomentsMean[:,0]-LeftKneeMomentsStd[:,0])*74.5/1000,(LeftKneeMomentsMean[:,0]+LeftKneeMomentsStd[:,0])*74.5/1000,alpha=0.7,color='salmon')
plt.title('Moment F/E Genou') # Poids 74.2 kg et le moment en N.mm/kg
plt.show()

## Moment du genou gauche non normalisé
plt.plot(range(101),RightKneeMomentsMean[:,0]*74.5/1000,'b')
plt.plot(range(101),(RightKneeMomentsMean[:,0]+RightKneeMomentsStd[:,0])*74.5/1000,'b--')
plt.plot(range(101),(RightKneeMomentsMean[:,0]-RightKneeMomentsStd[:,0])*74.5/1000,'b--')
plt.fill_between(range(101),(RightKneeMomentsMean[:,0]-RightKneeMomentsStd[:,0])*74.5/1000,(RightKneeMomentsMean[:,0]+RightKneeMomentsStd[:,0])*74.5/1000,alpha=0.7,color='skyblue')
plt.title('Moment F/E Genou') # Poids 74.2 kg et le moment en N.mm/kg
plt.show()


## EMG Biceps fémoral

LeftIJMean = acq.GetAnalog('Biceps Femoris Gauche').GetValues() + 2*acq.GetAnalog('Biceps Femoris Gauche').GetOffset()
LeftIJStd = acq1.GetAnalog('Biceps Femoris Gauche').GetValues() + 2*acq1.GetAnalog('Biceps Femoris Gauche').GetOffset()
RightIJMean = acq.GetAnalog('Biceps Femoris Droit').GetValues() + 2*acq.GetAnalog('Biceps Femoris Droit').GetOffset()
RightIJStd = acq1.GetAnalog('Biceps Femoris Droit').GetValues() + 2*acq1.GetAnalog('Biceps Femoris Droit').GetOffset()
plt.plot(range(101),LeftIJMean[:,0],'r')
plt.plot(range(101),LeftIJMean[:,0]+LeftIJStd[:,0],'r--')
plt.plot(range(101),LeftIJMean[:,0]-LeftIJStd[:,0],'r--')
plt.fill_between(range(101),LeftIJMean[:,0]-LeftIJStd[:,0],LeftIJMean[:,0]+LeftIJStd[:,0],alpha=0.7,color='salmon')
plt.plot(range(101),RightIJMean[:,0],'b')
plt.plot(range(101),RightIJMean[:,0]+RightIJStd[:,0],'b--')
plt.plot(range(101),RightIJMean[:,0]-RightIJStd[:,0],'b--')
plt.fill_between(range(101),RightIJMean[:,0]-RightIJStd[:,0],RightIJMean[:,0]+RightIJStd[:,0],alpha=0.7,color='skyblue')
plt.title('EMG IJ')
plt.show()

LeftVLMean = acq.GetAnalog('Vastus Lateralis Gauche').GetValues() + 2*acq.GetAnalog('Vastus Lateralis Gauche').GetOffset()
LeftVLStd = acq1.GetAnalog('Vastus Lateralis Gauche').GetValues() + 2*acq1.GetAnalog('Vastus Lateralis Gauche').GetOffset()
RightVLMean = acq.GetAnalog('Vastus Lateralis Droit').GetValues() + 2*acq.GetAnalog('Vastus Lateralis Droit').GetOffset()
RightVLStd = acq1.GetAnalog('Vastus Lateralis Droit').GetValues() + 2*acq1.GetAnalog('Vastus Lateralis Droit').GetOffset()
plt.plot(range(101),LeftVLMean[:,0],'r')
plt.plot(range(101),LeftVLMean[:,0]+LeftVLStd[:,0],'r--')
plt.plot(range(101),LeftVLMean[:,0]-LeftVLStd[:,0],'r--')
plt.fill_between(range(101),LeftVLMean[:,0]-LeftVLStd[:,0],LeftVLMean[:,0]+LeftVLStd[:,0],alpha=0.7,color='salmon')
plt.plot(range(101),RightVLMean[:,0],'b')
plt.plot(range(101),RightVLMean[:,0]+RightVLStd[:,0],'b--') # 'b--' courbe en pointillé bleu
plt.plot(range(101),RightVLMean[:,0]-RightVLStd[:,0],'b--')
plt.fill_between(range(101),RightVLMean[:,0]-RightVLStd[:,0],RightVLMean[:,0]+RightVLStd[:,0],alpha=0.7,color='skyblue')
plt.title('EMG VL')
plt.show()


## EMG: Rectus Femoris Gauche
LeftRFMean = acq.GetAnalog('Rectus Femoris Gauche').GetValues() + 2*acq.GetAnalog('Rectus Femoris Gauche').GetOffset() # Matrice de 101 valeur
LeftRFStd = acq1.GetAnalog('Rectus Femoris Gauche').GetValues() + 2*acq1.GetAnalog('Rectus Femoris Gauche').GetOffset()
plt.plot(range(101),LeftRFMean[:,0],'r') # 'r' ligne rouge continu
plt.plot(range(101),LeftRFMean[:,0]+LeftRFStd[:,0],'r--') # 'r--' en pointillé rouge
plt.plot(range(101),LeftRFMean[:,0]-LeftRFStd[:,0],'r--') #
plt.fill_between(range(101),LeftRFMean[:,0]-LeftRFStd[:,0],LeftRFMean[:,0]+LeftRFStd[:,0],alpha=0.7,color='salmon')
plt.show()


## EMG: Rectus Femoris Droit
RightRFMean = acq.GetAnalog('Rectus Femoris Droit').GetValues() + 2*acq.GetAnalog('Rectus Femoris Droit').GetOffset()
RightRFStd = acq1.GetAnalog('Rectus Femoris Droit').GetValues() + 2*acq1.GetAnalog('Rectus Femoris Droit').GetOffset()
plt.plot(range(101),RightRFMean[:,0],'b')
plt.plot(range(101),RightRFMean[:,0]+RightRFStd[:,0],'b--')
plt.plot(range(101),RightRFMean[:,0]-RightRFStd[:,0],'b--')
plt.fill_between(range(101),RightRFMean[:,0]-RightRFStd[:,0],RightRFMean[:,0]+RightRFStd[:,0],alpha=0.7,color='skyblue')
plt.title('EMG VL')
plt.show()


## Analyse dataset

## Pré-analyse
print (RightIJMean.shape) # Matrice 101 ligne et 1 colonne (Feature= Biceps Femoris)

print(RightRFMean.shape)# Matrice 101 lignes R

Biceps_Femoral= RightIJMean
Droit_Femoral=RightRFMean
Vaste_lateral=RightVLMean


EMG= np.c_[RightIJMean,RightVLMean] # Feature EMG Matrice 101 * 2

Moments_droit=np.delete(RightKneeMomentsMean,[1,2],1) # Couple en Nm (Target variable)

Moments_droit=np.multiply(Moments_droit, 10**-3)

print(Moments_droit.shape)

## Affichage de la base de donnée
plt.scatter(Droit_Femoral,Moments_droit)
plt.xlabel('EMG ')
plt.ylabel('Moment du genou')
plt.plot()
plt.show()

plt.scatter(Biceps_Femoral,Moments_droit)
plt.xlabel('EMG ')
plt.ylabel('Moment du genou')
plt.plot()
plt.show()

plt.scatter(Vaste_lateral,Moments_droit)
plt.xlabel('EMG ')
plt.ylabel('Moment du genou')
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




## On créé le vecteur theta + modèle
theta = np.random.randn(3,1) # [a,b,c]
print(theta)


def F_expo(X,theta): # F = exp(X * \theta)
    return np.exp(X.dot(theta))

def modele(X,theta):
    return X.dot(theta)


## Test modele sans minimisation
plt.scatter(Biceps_Femoral,Moments_droit)
plt.plot(Biceps_Femoral, F_expo(X_1,theta))
plt.xlabel('EMG ')
plt.ylabel('Moment du genou')
plt.plot()
plt.show()

## Fonction cout

def fonction_cout(X,y,theta):
    m= len(y)
    return 1/(2*m)*np.sum((F_expo(X,theta) - y)**2)


## Gradient

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(modele(X, theta) - y)


## Vérifier que la matrice dot(X.T,X) est inversible
print(alg.det(np.dot(X_3.T,X_3)))

## Equation normale
print(alg.inv(np.dot(X_3.T,X_3)))


theta = np.linalg.pinv(X_2.T.dot(X_2)).dot(X_2.T).dot(Moments_droit)



predictions = F_expo(X_2,theta)

plt.scatter(Droit_Femoral,Moments_droit)
plt.plot(Droit_Femoral, predictions, c='r')
plt.show()


## Descente du gradient
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    cost_history = np.zeros(n_iterations)

    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta) # mise a jour du parametre theta (formule du gradient descent)
        cost_history[i] = cost_function(X, y, theta) # on enregistre la valeur du Cout au tour i dans cost_history[i]

    return theta, cost_history

##
n_iterations = 1000
learning_rate = 0.01

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)

print(theta_final) # voici les parametres du modele une fois que la machine a été entrainée

# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = model(X, theta_final)

# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.show()



## Analyse du programme
def rcarre(y, pred):
    u = ((y - pred)**2).sum() #residu de la somme des carres
    v = ((y - y.mean())**2).sum()# somme totale des carres
    return 1 - u/v

print('R2=', rcarre(Moments_droit, predictions))

