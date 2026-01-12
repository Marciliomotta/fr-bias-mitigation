import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Configuracoes
DATASET_DIR = "dataset_treino"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

def carregar_dados():
    # Carrega dataset de TREINO
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        shuffle=True,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Carrega dataset de VALIDACAO
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        shuffle=False,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Captura nomes antes da otimizacao
    class_names = train_ds.class_names
    
    # Otimizacao para CPU
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names

def construir_modelo(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False 

    inputs = Input(shape=IMG_SIZE + (3,))

    # Data Augmentation EXTRA dentro do modelo (para dificultar a decoreba)
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, 
                    activation='softmax',
                    kernel_regularizer=l2(0.01))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plotar_historico(history):
    if not os.path.exists("logs_treinamento"):
        os.makedirs("logs_treinamento")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy Evolution')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Loss Evolution')
    
    plt.savefig("logs_treinamento/grafico_treino.png")
    print("Grafico salvo em logs_treinamento/")

def main():
    try:
        train_ds, val_ds, class_names = carregar_dados()
        print(f"Classes detectadas ({len(class_names)}): {class_names}")
        
        if len(class_names) != 20:
             print("AVISO: Numero de classes diferente de 20. Verifique o gerador de dados.")

        model = construir_modelo(len(class_names))
        
        checkpoint = ModelCheckpoint("modelos_salvos/modelo_mitigado_v1.keras", 
                                     monitor='val_accuracy', 
                                     save_best_only=True, 
                                     mode='max', 
                                     verbose=1)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print("\nIniciando treinamento (CPU)...")
        history = model.fit(train_ds, 
                            epochs=EPOCHS, 
                            validation_data=val_ds, 
                            callbacks=[checkpoint, early_stop])
        
        plotar_historico(history)
        print("\nTreinamento concluido. Modelo salvo em 'modelos_salvos/modelo_mitigado_v1.keras'")
        
    except Exception as e:
        print(f"\nErro Critico: {e}")

if __name__ == "__main__":
    main()