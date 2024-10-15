import torch 

def distance_matrix(x):
    # Calcule la matrice de distance euclidienne
    dist = torch.sum(x**2, 1).view(-1, 1) + torch.sum(x**2, 1).view(1, -1) - 2 * torch.mm(x, x.t())
    return torch.sqrt(dist)

def sum_k_nearest(dist_matrix, k=1):
    # Trie la matrice de distance et prend les k plus petites distances pour chaque point (en excluant la distance à lui-même qui est 0)
    _, indices = dist_matrix.sort(dim=1)
    k_nearest_indices = indices[:, 1:k+1]  # On exclut la première colonne (distance de 0 à lui-même)
    
    k_nearest_values = torch.gather(dist_matrix, 1, k_nearest_indices)
    sum_k_nearest = k_nearest_values.sum(dim=1)
    
    return sum_k_nearest

# Exemple
s = torch.randn((100, 10))  # vecteur s de dimension (n, d) où n=100 et d=10
print(s.shape)
dist_mat = distance_matrix(s)
print(dist_mat.shape)
result = sum_k_nearest(dist_mat, k=5)

print(result.shape)