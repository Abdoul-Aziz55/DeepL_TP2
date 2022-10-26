# DeepL_TP2

## Introduction

Le travail réalisé consiste à prédire la température d'une date sachant celle des 7 jours
précédents. Pour ce faire, deux modèles ont été utilisés: Le modèle RNN et le modèle CNN.

Le problème auquel on s'est intéréssé est alors de savoir entre les deux modèles, lequel est
le plus performant pour le type de données dont nous disposons.

## Expériences réalisées et paramètres

Pour faire la prédiction, des données météo de 2019 et de 2020 qui contiennent des températures
moyennes journalières on été utilisées.

Les données de 2019 ont servies de bases d'entrainement pour les modèles alors que les données de
2020 ont été utilisées comme données de test.

Lors des différentes expériences, notre attention s'est focalisée sur les loss d'entrainement et de validation
(mesurée lors des tests) au cours des différentes epochs d'entrainement.

Les paramètres suivants ont été communs aux deux expériences:

- 300 epochs d'entrainement => de test
- 0.001 de taux d'apprentissage
- Adam comme optimiseur
- Erreur des moindres carrées comme erreur

Plus particulièrement, on a les paramètres suivants par modèle:

#### RNN

Le modèle a été conçu avec 20 couches cachées linéaires

#### CNN

Le modèle est un modèle séquentiel qui combine une couche convolutionnelle à une dimension avec un noyau de taille 7
et un stride de 1 avec une fonction d'activation de type ReLU.

## Résultats et comparaison des deux modèles

### Résultats

![image](RNN2.png)
![image](CNN2.png)

### Comparaison

L'état de l'art montre que le modèle RNN est plus capable de traiter les problèmes dont les données sont séquentielles. Le CNN, quand à lui, est plus performant au niveau des données spatiales (les images).

1- Par rapport au temps :

On remarque que l'entrainement du RNN prend plus du temps par rapport à celui du CNN; [ par exemple le modèle CNN est entrainé pour une durée de 1.44 secondes alors que le RNN prenait jusqu'à 122 secondes.]

2- Par rapport à test loss :

Après avoir lancer pas mal de fois le code de CNN et celui de RNN; on remarque qu'avec le CNN on aboutit souvent à une valeur minimale de test loss inférieure à la valeur finale de test loss du RNN.
Avec le nombre d'époques (300) qu'on a utilisé on remarque également que pour le CNN , la test loss continue de diminuer alors que dans le cas du RNN , la courbe se stabilise au bout de 200 époques.

3-Les courbes Test Loss & Train Loss :

Parfois, en lancant l'entrainement du modèle de CNN , on avait comme résultat :
![image](CNN_debut.png)

Ce qui veut dire que le modèle n'apprenait presque rien. En relancant le code , on aboutit à des meilleurs résultats (voir l'image d'avant).

Alors qu'avec le RNN, on n'a pas eu ce cas.

## Conclusion de l'étude
