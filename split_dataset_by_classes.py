import os
import shutil

import numpy as np


def collect_areas_and_labels(original_dataset_dir, splits):
    """
    Percorre o dataset YOLO original (1 classe) e coleta:
      - Áreas (w*h) de todas as bounding boxes.
      - Tuplas (split, label_file) para cada arquivo de anotação, para reuso na segunda fase.
    Retorna (all_areas, all_labels).
    """
    all_areas = []
    all_labels = []

    for split in splits:
        labels_dir = os.path.join(original_dataset_dir, split, "labels")
        if not os.path.isdir(labels_dir):
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                # YOLO: class_id x_center y_center w h
                w = float(parts[3])
                h = float(parts[4])
                area = w * h
                all_areas.append(area)

            # Guardar para releitura na 2ª fase
            all_labels.append((split, label_file))

    return all_areas, all_labels


def compute_percentiles(areas, lower_percentile=33, upper_percentile=66):
    """
    Dado um array de áreas, computa e retorna os dois valores de corte (p33 e p66).
    Ajuste lower_percentile e upper_percentile conforme desejar.
    """
    p33, p66 = np.percentile(areas, [lower_percentile, upper_percentile])
    return p33, p66


def classify_bounding_box(area, p33, p66):
    """
    Retorna a categoria (small, medium, large) de acordo com os cortes p33 e p66.
    """
    if area < p33:
        return "small"
    elif area < p66:
        return "medium"
    else:
        return "large"


def create_output_structure(base_output_dir, dataset_names, splits):
    """
    Cria a estrutura de diretórios para cada dataset (small, medium, large).
    Retorna um dicionário {categoria: path}.
    """
    datasets_info = {}
    for category, folder_name in dataset_names.items():
        dataset_path = os.path.join(base_output_dir, folder_name)
        datasets_info[category] = dataset_path

        for split in splits:
            images_dir = os.path.join(dataset_path, split, "images")
            labels_dir = os.path.join(dataset_path, split, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

    return datasets_info


def split_dataset_by_size(original_dataset_dir, splits, all_labels, p33, p66, datasets_info, image_ext=".jpg"):
    """
    Lê novamente cada arquivo de label e classifica suas bounding boxes
    em small/medium/large, copiando as linhas e imagens para as pastas corretas.
    """
    # Para evitar copiar a mesma imagem repetidas vezes para uma mesma categoria
    copied_images = {
        "small": set(),
        "medium": set(),
        "large": set()
    }

    for (split, label_file) in all_labels:
        labels_dir = os.path.join(original_dataset_dir, split, "labels")
        images_dir = os.path.join(original_dataset_dir, split, "images")

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        lines_small = []
        lines_medium = []
        lines_large = []

        # Separa as BBs de acordo com o tamanho
        for line in lines:
            parts = line.strip().split()
            # class_id = int(parts[0]) # Sabemos que é 0, pois só há 1 classe
            w = float(parts[3])
            h = float(parts[4])
            area = w * h

            category = classify_bounding_box(area, p33, p66)

            # Renumera a classe para 0 (por convenção YOLO)
            new_line = "0 " + " ".join(parts[1:]) + "\n"

            if category == "small":
                lines_small.append(new_line)
            elif category == "medium":
                lines_medium.append(new_line)
            else:
                lines_large.append(new_line)

        # Copiar e gravar rótulos para cada categoria
        image_name = label_file.replace(".txt", image_ext)
        src_image_path = os.path.join(images_dir, image_name)

        if len(lines_small) > 0:
            dst_label_file = os.path.join(
                datasets_info["small"], split, "labels", label_file)
            with open(dst_label_file, "w") as f:
                f.writelines(lines_small)

            if (split, image_name) not in copied_images["small"]:
                copied_images["small"].add((split, image_name))
                if os.path.exists(src_image_path):
                    dst_image_file = os.path.join(
                        datasets_info["small"], split, "images", image_name)
                    shutil.copy2(src_image_path, dst_image_file)

        if len(lines_medium) > 0:
            dst_label_file = os.path.join(
                datasets_info["medium"], split, "labels", label_file)
            with open(dst_label_file, "w") as f:
                f.writelines(lines_medium)

            if (split, image_name) not in copied_images["medium"]:
                copied_images["medium"].add((split, image_name))
                if os.path.exists(src_image_path):
                    dst_image_file = os.path.join(
                        datasets_info["medium"], split, "images", image_name)
                    shutil.copy2(src_image_path, dst_image_file)

        if len(lines_large) > 0:
            dst_label_file = os.path.join(
                datasets_info["large"], split, "labels", label_file)
            with open(dst_label_file, "w") as f:
                f.writelines(lines_large)

            if (split, image_name) not in copied_images["large"]:
                copied_images["large"].add((split, image_name))
                if os.path.exists(src_image_path):
                    dst_image_file = os.path.join(
                        datasets_info["large"], split, "images", image_name)
                    shutil.copy2(src_image_path, dst_image_file)


def generate_data_yaml(dataset_dir, class_name):
    """
    Gera um arquivo data.yaml básico para um dataset YOLO de 1 classe.
    """
    train_images = os.path.join(dataset_dir, "train", "images")
    val_images = os.path.join(dataset_dir, "valid", "images")
    test_images = os.path.join(dataset_dir, "test", "images")

    content = f"""# data.yaml gerado automaticamente
train: {train_images}
val: {val_images}
test: {test_images}

nc: 1
names: [ "{class_name}_ship" ]
"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(content)


def main():
    # --------------------
    # CONFIGURAÇÕES
    # --------------------
    CWD = os.getcwd()
    def get_path(x): return os.path.join(CWD, x)
    ORIGINAL_DATASET_DIR = get_path("data/ships_v10i")  # Ajuste
    BASE_OUTPUT_DIR = get_path("data")  # Ajuste
    SPLITS = ["train", "valid", "test"]  # Ajuste se necessário
    IMAGE_EXT = ".jpg"  # Ajuste se for .png, .jpeg etc.

    # Pastas finais para cada categoria
    dataset_names = {
        "small":  "ships_categories_small",
        "medium": "ships_categories_medium",
        "large":  "ships_categories_large"
    }

    # -----------------------------------------------------
    # 2) COLETAR ÁREAS E DEFINIR PERCENTIS
    # -----------------------------------------------------
    print("Coletando bounding boxes do dataset original...")
    all_areas, all_labels = collect_areas_and_labels(
        ORIGINAL_DATASET_DIR, SPLITS)

    if len(all_areas) == 0:
        print("Nenhuma bounding box encontrada. Encerrando.")
        return

    p33, p66 = compute_percentiles(
        all_areas, lower_percentile=33, upper_percentile=66)
    print(f"Percentis calculados: p33={p33:.6f}, p66={p66:.6f}")

    # -----------------------------------------------------
    # 3) CRIAR ESTRUTURA DE PASTAS
    # -----------------------------------------------------
    datasets_info = create_output_structure(
        BASE_OUTPUT_DIR, dataset_names, SPLITS)

    # -----------------------------------------------------
    # 4) SEPARAR O DATASET POR TAMANHO
    # -----------------------------------------------------
    print("Classificando e criando datasets (small, medium, large)...")
    split_dataset_by_size(
        ORIGINAL_DATASET_DIR,
        SPLITS,
        all_labels,
        p33,
        p66,
        datasets_info,
        image_ext=IMAGE_EXT
    )

    # -----------------------------------------------------
    # 5) GERAR data.yaml PARA CADA CATEGORIA
    # -----------------------------------------------------
    print("Gerando arquivos data.yaml...")
    generate_data_yaml(datasets_info["small"],  "small")
    generate_data_yaml(datasets_info["medium"], "medium")
    generate_data_yaml(datasets_info["large"],  "large")

    print("Processo concluído com sucesso.")


if __name__ == "__main__":
    main()
