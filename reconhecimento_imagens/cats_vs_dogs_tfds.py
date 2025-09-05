# 🐱🐶 Classificação de Imagens: Cats vs Dogs (com TensorFlow Datasets)
# ===============================================================
# Este notebook foi adaptado para rodar no **VS Code + Windows 11 + Python 3.13.17**
# Usaremos o **TensorFlow Datasets (TFDS)** para baixar automaticamente o dataset Cats vs Dogs.
# Cada célula possui explicações didáticas em Markdown e comentários no código.

# ## 🔹 1. Preparação do Ambiente
# Execute este comando no terminal Bash do VS Code para instalar as dependências:
#
# ```bash
# pip install tensorflow tensorflow-datasets matplotlib numpy
# ```

# ## 🔹 2. Importação das Bibliotecas
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

print("✅ Bibliotecas importadas com sucesso!")

# ## 🔹 3. Carregar Dataset (TFDS baixa automaticamente)
# - O dataset Cats vs Dogs é baixado e armazenado localmente pelo TFDS.
# - Usamos `as_supervised=True` para retornar tuplas (imagem, rótulo).

(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

print("✅ Dataset carregado com sucesso!")
print(ds_info)

# ## 🔹 4. Pré-processamento das Imagens
# - Redimensionar imagens para 128x128
# - Normalizar pixels para valores entre 0 e 1
# - Criar lotes (batchs) para treino

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalização
    return image, label

ds_train = ds_train.map(preprocess).batch(BATCH_SIZE).shuffle(1000)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE)

print("✅ Pré-processamento concluído!")

# ## 🔹 5. Visualizar Amostras do Dataset
# Vamos exibir algumas imagens para verificar se o dataset foi carregado corretamente.

for images, labels in ds_train.take(1):
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy())
        plt.title("Cat" if labels[i].numpy()==0 else "Dog")
        plt.axis("off")
    plt.show()

# ## 🔹 6. Definição do Modelo CNN
# A rede neural convolucional será composta por:
# - Camadas Conv2D + MaxPooling para extrair características visuais
# - Flatten para achatar a matriz em vetor
# - Dense para classificação binária (gato ou cachorro)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Binária: gato ou cachorro
])

model.summary()

# ## 🔹 7. Compilação e Treinamento do Modelo
# - Otimizador: Adam (eficiente para CNNs)
# - Função de perda: Binary Crossentropy (classificação binária)
# - Métrica: Acurácia

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=5
)

# ## 🔹 8. Avaliação do Modelo
# Vamos analisar as curvas de acurácia e loss para entender o desempenho.

plt.figure(figsize=(12,5))

# Acurácia
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss')
plt.legend()

plt.show()

print("✅ Treinamento e avaliação concluídos!")
