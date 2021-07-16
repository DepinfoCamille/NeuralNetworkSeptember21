# FCN model
# when tuning start with learning rate->mini_self.batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np
import time 
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight


from utils.utils import save_logs
from utils.utils import calculate_metrics
import os
from collections import defaultdict
import json

CONV1D_DROPOUT_FRACTION = 0.3 # 0.1
DENSE_DROPOUT_FRACTION = 0.7 # 0.1
CONV1D_INITIAL_NB_FILTERS = 128
LEARNING_RATE = 0.00004#0.0001
KERNEL_CONV_L1 = 0#10e-8#None#regularizers.l1_l2()
KERNEL_CONV_L2 = 0#10e-8#None#regularizers.l1_l2()
KERNEL_DENSE_L1 = 0#0.0001#regularizers.l1_l2()
KERNEL_DENSE_L2 = 0#0.0001#regularizers.l1_l2()
BIAS_CONV= 0.01#0.01#regularizers.l2()
BIAS_DENSE = 0.01#0.01#regularizers.l2()



#https://medium.com/@matrixB/modified-cross-entropy-loss-for-multi-label-classification-with-class-a8afede21eb9


def weights(labels):
	'''labels: numpy array of shape (n_samples, nb_classes)'''

	N = labels.shape[0]
	class_weights = {}
	positive_weights = {}
	negative_weights = {}

	for index in range(labels.shape[1]):
		nb_ones = np.sum(labels[:, index])
		positive_weights[index] = N /(2 * nb_ones)
		negative_weights[index] = N /(2 * (N - nb_ones))

	class_weights['positive_weights'] = positive_weights
	class_weights['negative_weights'] = negative_weights
	return class_weights

def custom_loss_with_weights(Wp, Wn):

	def custom_loss(y_true, y_logit):
		'''
		Multi-label cross-entropy
		* Required "Wp", "Wn" as positive & negative class-weights
		y_true: true value
		y_logit: predicted value
		'''
		loss = float(0)
    
		for i, key in enumerate(Wp.keys()):
			print("i", i)
			first_term = Wp[key] * y_true[i] * K.log(y_logit[i] + K.epsilon())
			second_term = Wn[key] * (1 - y_true[i]) * K.log(1 - y_logit[i] + K.epsilon())
			loss -= (first_term + second_term)
			print("loss", loss)
		return loss


#https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

#https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
def get_weighted_loss(weights):
	def weighted_loss(y_true, y_pred):
		y_true = K.cast(y_true, "float32")
		return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
		#return K.mean(K.cast(weights[:,0]**(1-y_true), "float32")*K.cast(weights[:,1]**(y_true), "float32")* K.cast(K.binary_crossentropy(y_true, y_pred), "float32"), axis=-1)
	return weighted_loss

def f1_score(y_true, y_logit):
    '''
    Calculate F1 score
    y_true: true value
    y_logit: predicted value
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_logit, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    predicted_positives = K.sum(K.round(K.clip(y_logit, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return (2 * precision * recall) / (precision + recall + K.epsilon())

class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, y_train, \
				dropout_conv1D = CONV1D_DROPOUT_FRACTION, dropout_dense = DENSE_DROPOUT_FRACTION, \
				channels_conv1D = CONV1D_INITIAL_NB_FILTERS, \
				kernel_dense_l1 = KERNEL_DENSE_L1, kernel_dense_l2 = KERNEL_DENSE_L2, \
				bias_conv = BIAS_CONV, bias_dense = BIAS_DENSE, \
				batch_size = 16, verbose=False, build=True):
		self.output_directory = output_directory
		self.batch_size = batch_size
		if build == True:
			self.model = self.build_model(input_shape, nb_classes, y_train)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			#self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))
		return

	def build_model(self, input_shape, nb_classes, y_train):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=CONV1D_INITIAL_NB_FILTERS, kernel_size=8, padding='same', \
									kernel_regularizer=regularizers.l1_l2(l1 = kernel_conv_l1, l2 = kernel_conv_l2), \
									bias_regularizer=regularizers.l2(bias_conv))(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Dropout(CONV1D_DROPOUT_FRACTION)(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)


		#conv2 = keras.layers.LSTM(CONV1D_INITIAL_NB_FILTERS, 2*CONV1D_INITIAL_NB_FILTERS, dropout=0.5, recurrent_dropout=0.5)(conv1)
		conv2 = keras.layers.Conv1D(filters=2*CONV1D_INITIAL_NB_FILTERS, kernel_size=5, padding='same', \
									kernel_regularizer=regularizers.l1_l2(l1 = kernel_conv_l1, l2 = kernel_conv_l2), \
									bias_regularizer=regularizers.l2(bias_conv))(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Dropout(CONV1D_DROPOUT_FRACTION)(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(CONV1D_INITIAL_NB_FILTERS, kernel_size=3,padding='same', \
									kernel_regularizer=regularizers.l1_l2(l1 = kernel_conv_l1, l2 = kernel_conv_l2), \
									bias_regularizer=regularizers.l2(bias_conv))(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Dropout(CONV1D_DROPOUT_FRACTION)(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
		gap_layer = keras.layers.Dropout(DENSE_DROPOUT_FRACTION)(gap_layer)
		output_layer = keras.layers.Dense(nb_classes, activation='sigmoid', \
									kernel_regularizer=regularizers.l1_l2(l1 = kernel_dense_l1, l2 = kernel_dense_l2), \
									bias_regularizer=regularizers.l2(bias_dense))(gap_layer)
		#output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		# Add class weights
		#class_weights = weights(y_train)
		#Wp = class_weights['positive_weights']
		#Wn = class_weights['negative_weights']
		print("before_class_weights")
		self.class_weights = calculating_class_weights(y_train)
		print(y_train.shape)
		self.class_weights = self.class_weights.astype(np.float32)
		print("class weights", self.class_weights)

		#model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(), 
		model.compile(optimizer = keras.optimizers.Adam(), \
					 loss=get_weighted_loss(self.class_weights), metrics=['accuracy', f1_score])
					 #loss='binary_crossentropy', metrics=['accuracy', f1_score])
					  #loss = custom_loss_with_weights(Wp, Wn), metrics=['accuracy', f1_score])
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=LEARNING_RATE)

		file_path = os.path.join(self.output_directory, 'best_model.hdf5')

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr, model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		#self.batch_size = 16
		nb_epochs = 100#2000

		mini_batch_size = int(min(x_train.shape[0]/10, self.batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		y_pred = K.eval(self.model.predict(x_val))
		y_val = K.eval(y_val)
		y_pred_as_ones = np.zeros(y_pred.shape)
		print("y_pred.argmax(axis = 1)", y_pred.argmax(axis = 1))
		print("y_pred.argmax(axis = 1)", y_pred.argmax(axis = 1).shape)
		#y_pred_as_ones[:, y_pred.argmax(axis = 1)] = 1
		for i, arg in enumerate(y_pred.argmax(axis = 1)):
			y_pred_as_ones[i, arg] = 1

		print("pred", y_pred)
		print("pred as ones", y_pred_as_ones)
		print("true", y_val)
		print("diff", y_val - y_pred)
		print("error by class")
		error_by_class = defaultdict()
		for i in range(y_val.shape[1]):
			nb_errors = np.sum((y_val[:, i] > 0) & (y_pred_as_ones[:, i] < 1))
			total_class = np.sum(y_val[:, i] > 0)
			error_by_class[str(i)] = [nb_errors / total_class * 100, float(nb_errors), float(total_class)]
			print(nb_errors / total_class * 100, nb_errors, total_class)
		print("hist", hist)

		with open(os.path.join(self.output_directory, "error_by_class.json"), "w") as f:
			json.dump(error_by_class, f)


		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = os.path.join(self.output_directory, 'best_model.hdf5')
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred