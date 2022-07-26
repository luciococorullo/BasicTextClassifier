import shutil
import string
import tensorflow as tf
import os
import re

#* scarico il dataset e lo importo
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1",
                                  url,
                                  untar=True,
                                  cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, "train")
os.listdir(train_dir)
''' sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())
 '''

#* rimuovo le cartelle non utili dal dataset
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#* creo un validation set utilizzando una divisione 80:20 del training set
batch_size = 32
training_set = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=42)

validation_set = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=25)

testing_set = tf.keras.utils.text_dataset_from_directory('aclImdb/test',
                                                         batch_size=batch_size)


# definisco una funzione che si occupa della standardizzazione dei dati ovvero la pre elaborazione e formattazione del testo per rimuovere
# elementi come la punteggiatura o elementi HTML
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


#* creo un livello per il modello che si occupa di vettorializzare e standardizzare i dati di testo
#* si imposta l'output come intero in modo che crei indici interi per ogni token
max_features = 10000
sequence_lenght = 250

vect_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_lenght)

#* chiamo adapt per adattare lo stato del livello al set di dati in modo che il modello crei stringhe di numeri interi
train_text = training_set.map(lambda x, y: x)
vect_layer.adapt(train_text)


#definisco una gunzione per vedere il risultato del layer
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vect_layer(text), label


'''
text_batch, label_batch = next(iter(training_set))
first_rev, first_label = text_batch[0], label_batch[0]
print("Review ", first_rev)
print("Label", training_set.class_names[first_label])
print("Vect review", vectorize_text(first_rev, first_label))
 '''

#* applico il livello di vettoralizzazione a tutti i set
train_ds = training_set.map(vectorize_text)
val_ds = validation_set.map(vectorize_text)
test_ds = testing_set.map(vectorize_text)

# il metodo cache mette in cache i dati
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#* creo il modello
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        max_features + 1, embedding_dim
    ),  # prende le recensioni in int e cerca un vettore per ogni indice della parola
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(
    ),  # ritorna un vettore per ogni esempio facendo la media sulla dimensione della sequenza, in modo da gestire input a lunghezza variabile
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)  # strato denso con un solo nodo
])

#* creo la rete compilando il modello
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#* Eseguo il training del modello
history = model.fit(train_ds, validation_data=val_ds, epochs=15)

#* valuto il modello con il test set
loss, accuracy = model.evaluate(test_ds)
print("Loss", loss)
print("Accuracy", accuracy)

#* esporto il modello
export_model = tf.keras.Sequential(
    [vect_layer, model,
     tf.keras.layers.Activation('sigmoid')])

export_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=['accuracy'])

loss, accuracy = export_model.evaluate(testing_set)
print("Accuracy2", accuracy)
model.save('model/my_model')