
# 5 sequences of length 10
nb_sequences = 5
seq_length = 10
nb_classes = 20

input_shape = (seq_length,)
output_shape = (input_shape[0], nb_classes)

# uint8 is ok for <= 256 classes, otherwise use int32
input = Input(shape=input_shape, dtype='uint8')

# Without the output_shape, Keras tries to infer it using calling the function
# on an float32 input, which results in error in TensorFlow:
#   TypeError: DataType float32 for attr 'TI' not in list of allowed values: uint8, int32, int64

x_ohe = Lambda(K.one_hot,
               arguments={'nb_classes': nb_classes},
               output_shape=output_shape)(input)

x_classes = np.random.randint(0, nb_classes, size=(nb_sequences, seq_length))

