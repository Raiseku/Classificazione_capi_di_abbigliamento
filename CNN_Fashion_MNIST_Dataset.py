# -*- coding: utf-8 -*-


# Importo le librerie necessarie
from tensorflow import keras
import tensorflow as tf
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#Scarico il dataset fashion_mist direttamente dalla repository dei dataset della libreria Keras
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print("Il dataset è stato caricato correttamente.")

#Train_images contiene le immagini per il training della rete, ogni immagine è rappresentata da 28x28x1 pixel
print("Dimensione delle matrici contenenti le immagini del training set : ", train_images.shape[1],"x",train_images.shape[2]," e sono: ",train_images.shape[0])

#train_labels contiene le etichette corrette per ogni immagine del training set
print("Dimensione del vettore contenente le etichette di ogni immagine nel training set: ", train_labels.shape[0])

#test_images contiene le immagini per il testing della rete, ogni immagine è rappresentata da 28x28x1 pixel
print("Dimensione delle matrici contenenti le immagini del test set : ", test_images.shape[1],"x",test_images.shape[2]," e sono: ",test_images.shape[0])

#train_labels contiene le etichette corrette per ogni immagine del test set
print("Dimensione del vettore contenente le etichette di ogni immagine nel test set: ", test_labels.shape[0])

#print("Tutte le immagini hanno un solo canale, sono in bianco e nero.")

# Normalizzazione dei dati, tutti i valori delle matrici diventano tra 0 ed 1.
train_images = train_images / 255.0
test_images = test_images / 255.0
print("Normalizzazione avvenuta con successo")

print("Stampo le prime 25 immagini del dataset:")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

print("Creazione del modello in Keras:")

# Definisco il modello in maniera sequenziale. In questo modo potrò aggiungere livelli uno dietro l'altro tramite il metodo add()
model = tf.keras.Sequential()

# Per il primo livello inserisco 64 filtri di dimensione 3x3 con stride 1, cioè il filtro viene traslato di un passo alla volta.
# Inserendo padding = "same" si specifica la presenza di padding per fare in modo che l'output abbia la stessa dimensione dell'input
# Lavorando con immagini di dimensione 28x28x1, la stessa dimensione andrà impostata sul parametro input_shape
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", activation="relu", input_shape=(28,28,1)))
# Si ottiene il valore massimo in una finestra di 2x2 riducendo la dimensione del risultato prodotto dal precedente livello 
model.add(tf.keras.layers.MaxPooling2D(pool_size = 2))
# Imposto a 0 il 30% degli output che escono dal precedente livello per evitare overfitting della rete
model.add(tf.keras.layers.Dropout(0.3))

# Aggiungo un nuovo "blocco" ma questa volta il livello convolutivo ha 32 filtri
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding = "same", activation="relu" ))
model.add(tf.keras.layers.MaxPooling2D(pool_size = 2))
model.add(tf.keras.layers.Dropout(0.3))

# "Appiattisco" il risultato per passare da 2 dimensioni ad 1
model.add(tf.keras.layers.Flatten())

# Parte MLP della rete, aggiungo un livello fully connected con 256 neuroni e funzione d'attivazione relu
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

# Per l'ultimo livello saranno necessari 10 neuroni, uno per ogni classe, e funzione di attivazione softmax
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

# Compilo il modello con 'adam' per la gestione del learning rate.
# La loss function utilizzata è la sparse_categorical_crossentropy proprio perchè stiamo trattando un problema di classificazione.
# Nel caso in cui si utilizzasse la rappresentazione 'one-hot' per le classi, la loss function da utilizzare sarebbe stata la CategoricalCrossentropy in keras.
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',  metrics=["accuracy"])

# Stampo la topologia della rete con tutti i livelli utilizzati
plot_model(model,to_file="model_plot.png", show_shapes = True, show_layer_names = True)

# Effettuo il reshape delle immagini utilizzate per poterle utilizzare nell'addestramento della rete
train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

# Eseguo l'addestramento del modello passando le immagini per il training e quelle per il testing.
# Il numero di epoche scelto è 10 con un batch_size = 32 il quale indica che ogni 32 esempi verranno aggiornati i parametri della rete
history = model.fit(train_images, train_labels, epochs = 50, batch_size=32, validation_split = 0.2, verbose=1)

#Definisco la funzione per stampare le performance del modello
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5), dpi = 130)
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Accuracy durante il training e validation')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Accuratezza')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Loss durante il training e validation')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

# Mi salvo le previsioni che la rete genera sul test set chiamato test_images
y_pred = model.predict_classes(test_images)
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
target_names = ["Classe {} ({}) :".format(i,labels[i]) for i in range(10)]

# Stampo il report del modello, ottenendo i parametri Precision, Recall, F-score ed Accuracy per la valutazione
print(classification_report(test_labels, y_pred, target_names = target_names))

# Stampo la matrice di confusione per vedere tutte le previsioni fatte dal modello addestrato
print("Matrice di confusione:")
cm=confusion_matrix(test_labels,y_pred)
print(cm)