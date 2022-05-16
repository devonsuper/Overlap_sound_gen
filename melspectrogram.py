import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras

"""

The code for the layer isn't mine, but this should convert a tensor that was generated from a sound file into a mel spectrogram.
(mel spectrograms are spectrograms that are scaled to represent sound in the way that humans hear it. This format was chosen to 
	incentivize the model to focus on the parts of sound that matter more to the human ear.)

"""


# assuming num_fft = 512
NUM_FFT = 512
NUM_FREQS = 257
# some tentative constants
NUM_MEL = 60
SAMPLE_RATE = 44100
F_MIN = 0
F_MAX = 12000

class LogMelgramLayer(tf.keras.layers.Layer):
	def __init__(self, num_fft, hop_length, **kwargs):
		super(LogMelgramLayer, self).__init__(**kwargs)
		self.num_fft = num_fft
		self.hop_length = hop_length

		assert num_fft // 2 + 1 == NUM_FREQS
		lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
			num_mel_bins=NUM_MEL,
			num_spectrogram_bins=NUM_FREQS,
			sample_rate=SAMPLE_RATE,
			lower_edge_hertz=F_MIN,
			upper_edge_hertz=F_MAX,
		)

		self.lin_to_mel_matrix = lin_to_mel_matrix

	def build(self, input_shape):
		self.non_trainable_weights.append(self.lin_to_mel_matrix)
		super(LogMelgramLayer, self).build(input_shape)

	def call(self, input):
		"""
		Args:
			input (tensor): Batch of mono waveform, shape: (None, N)
		Returns:
			log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
		"""

		def _tf_log10(x):
			numerator = tf.math.log(x)
			denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
			return numerator / denominator
	  
		# tf.signal.stft seems to be applied along the last axis
		stfts = tf.signal.stft(
			input, frame_length=self.num_fft, frame_step=self.hop_length
		)
		mag_stfts = tf.abs(stfts)

		melgrams = tf.tensordot(tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0])
		log_melgrams = _tf_log10(melgrams + tf.keras.backend.epsilon())
		return tf.expand_dims(log_melgrams, 3)

	def get_config(self):
		config = {'num_fft': self.num_fft, 'hop_length': self.hop_length}
		base_config = super(LogMelgramLayer, self).get_config()
		return dict(list(config.items()) + list(base_config.items()))