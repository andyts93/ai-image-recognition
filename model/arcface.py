import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFace(nn.Module):
    """
    Implementazione del layer ArcFace.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Crea i pesi del classificatore, che rappresentano i centri delle classi
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Valori costanti per i calcoli matematici
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        # 1. Normalizza l'embedding e i pesi (centri delle classi)
        embedding = F.normalize(embedding)
        weight = F.normalize(self.weight)

        # 2. Calcola il coseno della similarità tra embedding e centri delle classi
        cosine = F.linear(embedding, weight)

        # 3. Applica il margine angolare (la "magia" di ArcFace)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Se l'angolo è troppo grande, usa una formula di approssimazione
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4. Crea il vettore di output
        # Prendi il one-hot encoding delle etichette corrette
        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Inserisci i valori di 'phi' solo per le classi corrette
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 5. Scala l'output per la loss
        output *= self.s

        return output