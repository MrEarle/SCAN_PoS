# PoS SCAN

Experiments that include Part of Speech to SCAN dataset.

Integrante | Correo
---|---
Benjamin Earle | biearle@uc.cl
Jorge Perez | jiperez11@uc.cl

## Data

En el archivo `src/data/scan.py` se encuentra el código para preparar el dataset. En particular, se utilizan 2 funciones:

- `get_dataset(experiment: str)`:
  - El argumento que toma es el split a usar de scan. Se usan 3 en particular:
    - `simple`: Es el baseline de SCAN. Tanto el entrenamiento como el test son i.i.d.
    - `addprim_jump`: Es el experimento que se agrega la traducción de composiciones de "jump" al momento de test. En entrenamiento, solo ve "jump" sin modificaciones.
    - `mcd1`: Es un split realizado por un paper posterior. Busca maximizar la necesidad de generalización para resolver la tarea.
  - Esta función preprocesa el dataset para entregarlo con el siguiente formato para cada item: `(x, p, y)`.
    - `x`: El input de scan. Es una versión codificada de una frase como "jump left and run twice".
      - La codificación corresponde a una lista de los índices correspondiente a cada palabra en el vocabulario. Esta lista es de largo `MAX_IN_SEQUENCE_LENGTH` (12), por lo que cada frase puede tener a lo mas 12 palabras, incluyendo `<sos>` y `<eos>`.
    - `p`: Part of Speech. Es una versión codificada de la secuencia de part of speech para una frase. Para el ejemplo de `x`, sería la codificación de "VRB DIR CNJ VRB ADV".
      - La codificación es la misma que para `x`, pero con el vocabulario de parts of speech.
    - `y`: Output de scan. Corresponde a una versión codificada de una secuencia de salida. Para el ejemplo de `x` sería la codificación de "TURN_LEFT JUMP RUN RUN".
      - La codificación corresponde a la misma que en `x`, solo que con el vocabulario de salida, y con un largo máximo de `MAX_OUT_SEQUENCE_LENGTH` (50).
  - Retorno: `train_dataset, test_dataset, (in_vectorizer, pos_vectorizer, out_vectorizer)`
    - `train_dataset`: Dataset de entrenamiento con el formato descrito. Ya está cacheado y con prefetch.
    - `test_dataset`: Dataset de test o validación. Ya está cacheado y con prefetch.
    - `*_vectorizer`: Son los `TextVectorizer` de Keras para procesar el texto.
- `get_final_map_function(include_pos_tag: bool, teacher_forcing: float)`: Funcion que toma el dataset, y lo mapea a un formato más cómodo para hacer los experimentos. Retorna 2 diccionarios, uno para input y otro para output.
  - Diccionario de input: Contiene las siguientes llaves y valores.
    - `input_embedding`: Corresponde al `x` de `get_dataset`.
    - `pos_embedding` (Opcional): Corresponde al `p` de `get_dataset`. Solo se incluye si `include_pos_tag == "input"`, es decir, si se realizará el entrenamiento con el PoS como input del modelo.
    - `action_input` (Opcional): Se incluye si `teacher_forcing > 0`. Esto es para cuando se quiere entrenar con teacher forcing. Corresponde a las acciones ground truth para entrenar el seq2seq. Es decir, el `y` de `get_dataset`
  - Diccionario de output: Contiene las siguientes llaves y valores.
    - `action_output`: Corresponde al ground truth de las acciones, para usar como supervisión. Es la conversión a *onehot* del `y` de `get_dataset`
    - `pos_output` (Opcional): Corresponde al ground truth del Part of Speech, para usar como supervision de la tarea auxiliar. Es la conversión a *onehot* del `p` de `get_dataset`.

Modo de uso:

```py
train_dataset, test_dataset, _ = get_dataset("addprim_jump")

final_map_fn = get_final_map_function(True, 0.5)


train_dataset = train_dataset.map(final_map_fn)
test_dataset = test_dataset.map(final_map_fn)

model.fit(train_dataset.shuffle(1000, reshuffle_each_iteration=True).batch(512))
```

Para un ejemplo de como compilar y entrenar el modelo, ver `src/tasks/train.py`

Notar que el modelo debe recibir como primer argumento el diccionario de input, y debe retornar un diccionario con las mismas llaves que el de output, solo que con las predicciones realizadas.

# Compilar loss y accuracy para el modelo multi output

Ejemplo de `src/tasks/train.py`:

```py
# Create metrics and losses according to task

# pad_idx es el pad id que creo el vectorizador de texto.
# mask_value es una mascara para que el modelo no sobreajuste a solo predecir los valores de padding.
mask_value: tf.Tensor = tf.one_hot(pad_idx, OUT_VOCAB_SIZE)

# Categorical crossentropy, pero solo se aplica para lo que no corresponde a padding. Este viene de src/utils/loss
masked_categorical_crossentropy = get_masked_categorical_crossentropy(mask_value)

# El diccionario de losses tiene que tener las mismas llaves que el diccionario de output del modelo. Asi Tensorflow sabe automaticamente como calcular las perdidas cuando hay multiples outputs
losses = {ACTION_OUTPUT_NAME: masked_categorical_crossentropy}
metrics = {
    # MaskedCategoricalAccuracy es similar al crosentropy anterior.
    ACTION_OUTPUT_NAME: MaskedCategoricalAccuracy(
        mask_value, name="accuracy" if params["include_pos_tag"] == "aux" else f"{ACTION_OUTPUT_NAME}_accuracy"
    )
}

# Si vamos a usar PoS como tarea auxiliar, hay que entregar loss y accuracy para este output extra del modelo. Es similar a lo anterior, solo que el pad_idx debe ser el correspondiente al vocabulario de PoS.
if params["include_pos_tag"] == "aux":
    pos_mask_value: tf.Tensor = tf.one_hot(pad_idx, POS_VOCAB_SIZE)
    losses[POS_OUTPUT_NAME] = get_masked_categorical_crossentropy(pos_mask_value)
    metrics[POS_OUTPUT_NAME] = MaskedCategoricalAccuracy(pos_mask_value, name="accuracy")

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=5.0)
model.compile(optimizer=optimizer, loss=losses, metrics=metrics, run_eagerly=True)
  ```
  
## Transformer

En el archivo TransformerPosTag.ipynb se encuentra el código para crear un modelo transformer, obtener los datos y entrenar el modelo. Se entregan 2 loops de entrenamiento, en donde 1 es con pos tag como tarea auxiliar y el otro es sin pos tag.
