import tensorflow as tf
import tensorflow_io as tfio
import melspectrogram
import tensorflow.keras as keras

from melspectrogram import LogMelgramLayer, NUM_FFT

"""
converts the end result coefficients into a mel spectrogram representing what sound the input coefficients would create when combined with the resource sounds
"""


def numtospectrogramfunc(sounds):
	class numtospectrogram(keras.layers.Layer):
		def __init__(self, input_dim):
			super(numtospectrogram, self).__init__()
			self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)
	
		def call(self, inputs):
			self.svals = tf.transpose(inputs)
			#print(inputs.shape)
			self.svals *= tf.stack(sounds)#tf.transpose()
			#TODO this may be causing issues. It _should_ add up all of the sounds into one spectrogram
			self.merged = tf.reduce_sum(self.svals, axis=[0])
	
			lml = LogMelgramLayer(num_fft=NUM_FFT, hop_length=256)
	
			fspec = lml(tf.expand_dims(self.merged, axis=0))
			print(fspec)

			return fspec

	return numtospectrogram