import os
import shutil

import pandas as pd


def create_class_specific_datasets(input_dir, output_base_dir):
    """
    Cria clones do dataset YOLO, onde cada clone contém apenas uma classe.

    Parameters:
        input_dir (str): Diretório base do dataset original no formato YOLO.
        output_base_dir (str): Diretório base para os clones do dataset separados por classe.
    """
    subsets = ["train", "valid", "test"]

    for subset in subsets:
        # Caminhos para imagens e rótulos
        input_images_dir = os.path.join(input_dir, subset, "images")
        input_labels_dir = os.path.join(input_dir, subset, "labels")

        # Verificar se os diretórios existem
        if not os.path.exists(input_images_dir) or not os.path.exists(input_labels_dir):
            print(f"Subdiretório {subset} não encontrado, pulando...")
            continue

        # Criar datasets separados por classe
        for label_file in os.listdir(input_labels_dir):
            label_path = os.path.join(input_labels_dir, label_file)

            # Verificar se o arquivo é de rótulo
            if not os.path.isfile(label_path) or not label_path.endswith(".txt"):
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            # Processar rótulos e separar por classe
            class_dict = {}
            for line in lines:
                cls, x_center, y_center, width, height = line.strip().split()
                if cls not in class_dict:
                    class_dict[cls] = []
                class_dict[cls].append(f"{cls} {x_center} {
                                       y_center} {width} {height}")

            # Criar datasets para cada classe
            for cls, class_lines in class_dict.items():
                output_images_dir = os.path.join(
                    output_base_dir, f"class_{cls}", subset, "images")
                output_labels_dir = os.path.join(
                    output_base_dir, f"class_{cls}", subset, "labels")
                os.makedirs(output_images_dir, exist_ok=True)
                os.makedirs(output_labels_dir, exist_ok=True)

                # Copiar a imagem correspondente para o dataset da classe
                image_file = label_file.replace(".txt", ".jpg")
                input_image_path = os.path.join(input_images_dir, image_file)
                output_image_path = os.path.join(output_images_dir, image_file)
                if os.path.exists(input_image_path):
                    shutil.copy(input_image_path, output_image_path)

                # Salvar o rótulo no dataset da classe
                output_label_path = os.path.join(output_labels_dir, label_file)
                with open(output_label_path, "w") as f:
                    f.write("\n".join(class_lines) + "\n")

    print(f"Datasets separados por classe salvos em: {output_base_dir}")


# Caminhos
# Substitua pelo caminho base do dataset original
base_dir = "data/ships_categories"
# Substitua pelo caminho base de saída
output_base_dir = "data/ships_by_class"

# Criar os datasets separados por classe
create_class_specific_datasets(base_dir, output_base_dir)
