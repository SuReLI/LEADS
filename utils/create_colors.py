import matplotlib.pyplot as plt

def get_n_distinct_colors_deterministic(n):
  """
  Retourne les n couleurs les plus distinctes possibles de manière déterministe.

  Args:
    n: Le nombre de couleurs à retourner.

  Returns:
    Une liste de n couleurs.
  """

  # Initialise une liste de couleurs.
  colors = []

  # Génère des couleurs en utilisant le système de couleurs RGB.
  for i in range(n):
    r = i % 256
    g = (i // 256) % 256
    b = (i // (256 * 256)) % 256
    color = (r, g, b)

    # Ajoute la couleur à la liste si elle est distincte des autres couleurs de la liste.
    if all(color != existing_color for existing_color in colors):
      colors.append(color)

  return colors


if __name__ == "__main__":
  # Demande à l'utilisateur de saisir le nombre de couleurs souhaité.
  n = input("Combien de couleurs souhaitez-vous ? ")

  # Convertit la saisie de l'utilisateur en un entier.
  n = int(n)

  # Obtient les n couleurs les plus distinctes possibles.
  colors = get_n_distinct_colors_deterministic(n)

  # Affiche les couleurs.
  print(colors)
