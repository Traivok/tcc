import os
import shutil

import numpy as np
import pandas as pd
import yaml


def categorize_yolo_labels_in_directory(input_dir, output_dir, percentiles=[25, 50, 75]):
    """
    Processa todos os arquivos YOLO em um diretório, reatribui classes com base no tamanho 
    dos bounding boxes e salva os novos arquivos.

    Parameters:
        input_dir (str): Diretório contendo os arquivos de rótulos no formato YOLO.
        output_dir (str): Diretório onde os arquivos categorizados serão salvos.
        percentiles (list): Percentis para categorizar os tamanhos.

    """
    # Garantir que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    all_bboxes = []

    # Coletar todos os bounding boxes para calcular percentis globais
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if os.path.isfile(input_path) and input_path.endswith(".txt"):
            with open(input_path, "r") as f:
                for line in f:
                    cls, x_center, y_center, width, height = map(
                        float, line.strip().split())
                    all_bboxes.append(
                        [file_name, cls, x_center, y_center, width, height])

    # Criar DataFrame com todos os bounding boxes
    columns = ["file_name", "class", "x_center", "y_center", "width", "height"]
    df = pd.DataFrame(all_bboxes, columns=columns)

    # Calcular os percentis globais
    width_percentiles = np.percentile(df['width'], percentiles)
    height_percentiles = np.percentile(df['height'], percentiles)

    # Definir categorias
    categories = ['mini', 'small', 'medium', 'large']
    category_mapping = {cat: i for i, cat in enumerate(categories)}

    def assign_category(width, height):
        for i, (w_p, h_p) in enumerate(zip(width_percentiles, height_percentiles)):
            if width <= w_p and height <= h_p:
                return category_mapping[categories[i]]
        return category_mapping[categories[-1]]  # Categoria 'large'

    # Atribuir nova classe
    df['new_class'] = df.apply(
        lambda row: assign_category(row['width'], row['height']), axis=1
    )

    # Salvar arquivos categorizados
    for file_name, group in df.groupby('file_name'):
        output_path = os.path.join(output_dir, file_name)
        group[['new_class', 'x_center', 'y_center', 'width', 'height']].to_csv(
            output_path, sep=' ', header=False, index=False
        )

    print(f"Arquivos categorizados salvos em: {output_dir}")


def copy_images_and_labels(base_dir, output_base_dir):
    """
    Copia os diretórios de imagens e processa os rótulos.

    Parameters:
        base_dir (str): Caminho base do dataset original.
        output_base_dir (str): Caminho base do dataset categorizado.
    """
    subsets = ["train", "valid", "test"]

    for subset in subsets:
        # Caminhos de entrada e saída para imagens e rótulos
        input_images_dir = os.path.join(base_dir, subset, "images")
        input_labels_dir = os.path.join(base_dir, subset, "labels")
        output_images_dir = os.path.join(output_base_dir, subset, "images")
        output_labels_dir = os.path.join(output_base_dir, subset, "labels")

        # Copiar imagens
        os.makedirs(output_images_dir, exist_ok=True)
        shutil.copytree(input_images_dir, output_images_dir,
                        dirs_exist_ok=True)

        # Processar e categorizar rótulos
        categorize_yolo_labels_in_directory(
            input_labels_dir, output_labels_dir)

    print(f"Imagens e rótulos categorizados copiados para: {output_base_dir}")


def update_data_yaml(base_dir, output_base_dir, categories):
    """
    Atualiza o arquivo data.yaml com o número de classes e os nomes das novas classes.

    Parameters:
        base_dir (str): Caminho base do dataset original.
        output_base_dir (str): Caminho base do dataset categorizado.
        categories (list): Lista com os nomes das categorias.
    """
    input_yaml_path = os.path.join(base_dir, "data.yaml")
    output_yaml_path = os.path.join(output_base_dir, "data.yaml")

    # Carregar o YAML original
    with open(input_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    # Atualizar `nc` e `names`
    data_yaml["nc"] = len(categories)
    data_yaml["names"] = categories

    # Ajustar os caminhos relativos
    data_yaml["train"] = os.path.relpath(
        os.path.join(output_base_dir, "train", "images"))
    data_yaml["val"] = os.path.relpath(
        os.path.join(output_base_dir, "valid", "images"))
    data_yaml["test"] = os.path.relpath(
        os.path.join(output_base_dir, "test", "images"))

    # Salvar o novo YAML
    with open(output_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"Arquivo data.yaml atualizado salvo em: {output_yaml_path}")


# Caminhos
# Substitua pelo caminho base do dataset original
base_dir = "data/ships_v10i"
# Substitua pelo caminho base de saída
output_base_dir = "data/ships_categories"

# Categorias e nomes
categories = ['mini', 'small', 'medium', 'large']

# Copiar imagens, processar rótulos e atualizar data.yaml
copy_images_and_labels(base_dir, output_base_dir)
update_data_yaml(base_dir, output_base_dir, categories)
