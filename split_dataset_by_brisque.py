import math
import os
import shutil

import cv2
# Evita problemas de backend ao salvar plots em ambiente sem interface
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brisque import BRISQUE
from tqdm import tqdm

matplotlib.use('Agg')  # 'Agg' permite salvar gráficos sem mostrar em tela


def calculate_brisque_score(image_bgr):
    """
    Calcula o BRISQUE de uma imagem no formato BGR (OpenCV).
    """
    # Converter BGR para RGB, pois a lib brisque espera RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    brisque_obj = BRISQUE(url=False)
    score = brisque_obj.score(image_rgb)
    return score


def collect_brisque_scores_dir(original_dataset_dir, splits, image_ext=".jpg"):
    file_brisque_pairs = []

    for split in splits:
        images_dir = os.path.join(original_dataset_dir, split, "images")
        if not os.path.isdir(images_dir):
            print(f"[AVISO] Não existe diretório de imagens em: {images_dir}")
            continue

        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(image_ext)
        ]

        print(f"[{split}] Calculando BRISQUE para {len(image_files)} imagens...")
        for img_name in tqdm(image_files, ncols=80):
            img_path = os.path.join(images_dir, img_name)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"[ERRO] Não foi possível abrir: {img_name}")
                continue

            score = calculate_brisque_score(image_bgr)
            if math.isnan(score):
                print(
                    f"[AVISO] BRISQUE retornou NaN para {img_name}. Definindo como 999.")
                score = 999.0  # ou continue se quiser pular

            file_brisque_pairs.append([img_name, score, split])

    return file_brisque_pairs


def compute_percentiles_brisque(brisque_values, lower_percentile=33, upper_percentile=66):
    """
    Dado um array de brisque_values, computa e retorna p33 e p66.
    """
    p33, p66 = np.percentile(
        brisque_values, [lower_percentile, upper_percentile])
    return p33, p66


def classify_by_brisque(score, p33, p66):
    """
    Retorna a categoria (good, medium, bad) de acordo com p33 e p66.
    Quanto menor o score de BRISQUE, melhor a imagem.
    """
    if score < p66:
        return "good"
    else:
        return "bad"


def create_output_structure(base_output_dir, dataset_names, splits):
    """
    Cria a estrutura de diretórios para cada dataset (good, medium, bad).
    Mantém subpastas de train, valid, test.
    Retorna {categoria: path}.
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


def generate_data_yaml(dataset_dir, class_name):
    """
    Gera um arquivo data.yaml básico para um dataset YOLO.
    (Se não usar YOLO, pode remover essa parte.)
    """
    train_images = os.path.join(dataset_dir, "train", "images")
    val_images = os.path.join(dataset_dir, "valid", "images")
    test_images = os.path.join(dataset_dir, "test", "images")

    content = f"""# data.yaml gerado automaticamente
train: {train_images}
val: {val_images}
test: {test_images}

nc: 1
names: [ "{class_name}_images" ]
"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(content)


def copy_images_by_brisque(
    original_dataset_dir,
    splits,
    file_brisque_pairs,
    p33,
    p66,
    datasets_info,
    image_ext=".jpg"
):
    """
    Classifica cada imagem em (good/medium/bad) e copia para
    a estrutura de pastas do respectivo conjunto (train/valid/test).
    Retorna a contagem de quantas foram para cada categoria.
    """
    # Evitar copiar a mesma imagem mais de uma vez
    copied_images = {
        "good": set(),
        "medium": set(),
        "bad": set()
    }

    for filename, brisque_score, split in file_brisque_pairs:
        score_cat = classify_by_brisque(brisque_score, p33, p66)

        src_image_path = os.path.join(
            original_dataset_dir, split, "images", filename)
        src_label_path = os.path.join(
            original_dataset_dir, split, "labels", filename.replace(
                image_ext, ".txt")
        )

        if (split, filename) not in copied_images[score_cat]:
            copied_images[score_cat].add((split, filename))

            # Copiar a imagem
            if os.path.exists(src_image_path):
                dst_image_file = os.path.join(
                    datasets_info[score_cat], split, "images", filename
                )
                shutil.copy2(src_image_path, dst_image_file)

            # Copiar label (opcional)
            if os.path.exists(src_label_path):
                dst_label_file = os.path.join(
                    datasets_info[score_cat], split, "labels", filename.replace(
                        image_ext, ".txt")
                )
                shutil.copy2(src_label_path, dst_label_file)

    # Retornar quantas imagens foram para cada categoria
    return {
        "good":   len(copied_images["good"]),
        "medium": len(copied_images["medium"]),
        "bad":    len(copied_images["bad"])
    }


def main():
    # ------------------------------------------------------
    # 1) CONFIGURAÇÕES
    # ------------------------------------------------------
    CWD = os.getcwd()
    def get_path(x): return os.path.join(CWD, x)

    # Ajuste para o seu dataset
    ORIGINAL_DATASET_DIR = get_path("data/ships_v10i")
    BASE_OUTPUT_DIR = get_path("data")
    CSV_SCORES_PATH = get_path("data/brisque_scores_calculados.csv")

    # Splits do seu dataset YOLO
    SPLITS = ["train", "valid", "test"]
    IMAGE_EXT = ".jpg"

    # Pasta de saída para cada faixa de BRISQUE
    dataset_names = {
        "good":   "brisque_good",
        "medium": "brisque_medium",
        "bad":    "brisque_bad"
    }

    # ------------------------------------------------------
    # (A) CARREGAR OU CALCULAR O BRISQUE
    # ------------------------------------------------------
    if os.path.exists(CSV_SCORES_PATH):
        print("[INFO] CSV de BRISQUE já existe. Carregando:", CSV_SCORES_PATH)
        df_scores = pd.read_csv(CSV_SCORES_PATH)
        file_brisque_pairs = df_scores[[
            'filename', 'brisque', 'split']].values.tolist()
    else:
        print("[INFO] Calculando BRISQUE para cada imagem...")
        file_brisque_pairs = collect_brisque_scores_dir(
            ORIGINAL_DATASET_DIR,
            SPLITS,
            image_ext=IMAGE_EXT
        )
        # Salva no CSV
        if len(file_brisque_pairs) > 0:
            df_scores = pd.DataFrame(file_brisque_pairs, columns=[
                                     "filename", "brisque", "split"])
            df_scores.to_csv(CSV_SCORES_PATH, index=False)
            print(f"[INFO] CSV criado em: {CSV_SCORES_PATH}")
        else:
            print("[ERRO] Não foram encontradas imagens para cálculo.")
            return

    # ------------------------------------------------------
    # (B) PLOTAR HISTOGRAMA E SALVAR
    # ------------------------------------------------------
    all_brisque_values = [item[1] for item in file_brisque_pairs]

    # Gerar histograma e salvar como PNG
    plt.figure(figsize=(8, 5))
    plt.hist(all_brisque_values, bins=30, color='steelblue', alpha=0.7)
    plt.title("Histograma de BRISQUE")
    plt.xlabel("BRISQUE Score")
    plt.ylabel("Frequência")
    hist_path = get_path("data/brisque_hist.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"[INFO] Histograma salvo em: {hist_path}")

    # ------------------------------------------------------
    # (C) DEFINIR PERCENTIS E CRIAR DATASET
    # ------------------------------------------------------
    p33, p66 = compute_percentiles_brisque(all_brisque_values, 33, 66)
    print(f"[INFO] p33={p33:.2f}, p66={p66:.2f}")

    # Criar estrutura de pastas
    datasets_info = create_output_structure(
        BASE_OUTPUT_DIR, dataset_names, SPLITS)

    # Copiar imagens
    category_counts = copy_images_by_brisque(
        ORIGINAL_DATASET_DIR,
        SPLITS,
        file_brisque_pairs,
        p33,
        p66,
        datasets_info,
        image_ext=IMAGE_EXT
    )

    # Mostrar como ficou a subdivisão
    total_imgs = sum(category_counts.values())
    print("\n[INFO] Divisão do dataset:")
    print(f"  GOOD:   {category_counts['good']} imagens")
    print(f"  MEDIUM: {category_counts['medium']} imagens")
    print(f"  BAD:    {category_counts['bad']} imagens")
    print(f"  TOTAL:  {total_imgs} imagens")

    # (Opcional) data.yaml para cada faixa
    print("\nGerando arquivos data.yaml...")
    generate_data_yaml(datasets_info["good"],   "good")
    # generate_data_yaml(datasets_info["medium"], "medium")
    generate_data_yaml(datasets_info["bad"],    "bad")

    print("\nProcesso concluído com sucesso!")


if __name__ == "__main__":
    main()
