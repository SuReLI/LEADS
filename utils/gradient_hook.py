import torch

# Limite pour le tensor
LIMITE = 5.0

# Initialisation du tensor avec requires_grad=True pour qu'il puisse être mis à jour via des gradients
x = torch.tensor([4.0], requires_grad=True)

# Définir le gradient hook
def clamp_gradient(grad):
    # Si la valeur de x dépasse la LIMITE, mettre le gradient à 0
    if x.item() > LIMITE:
        return torch.zeros_like(grad)
    return grad

# Enregistrer le hook
hook = x.register_hook(clamp_gradient)

# Fonction de perte fictive
loss = x * x

# Backpropagation
loss.backward()

# Mise à jour de x en utilisant un taux d'apprentissage simple
learning_rate = 0.1
with torch.no_grad():
    x -= learning_rate * x.grad

print(x)  # Afficher la nouvelle valeur de x

# N'oubliez pas de supprimer le hook si vous n'en avez plus besoin
hook.remove()
