import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras

from melspectrogram import LogMelgramLayer, NUM_FFT
from numtospectrogram import numtospectrogramfunc

from os import walk, environ

from keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten


environ["CUDA_VISIBLE_DEVICES"] = "-1"

#training sounds
soundpath = 'D:/Programs/archive/Machine Learning and AI/overlapcreation/trainingsounds/'

#resource sounds
resourcesoundpath = "D:/Programs/archive/Machine Learning and AI/overlapcreation/resourcesounds/"
resourcesoundfiles = ["".join(resourcesoundpath + i) for i in list(walk(resourcesoundpath))[0][2]]

numresourcesounds = 66

resourcesoundtensor = None

soundlength = 8000


#Convert flac files into tensors for training
def soundFilesToTensor(paths):
	soundlist = []
	for path in paths:
		if (path.split(".")[1] == "flac"):
			print(path)
			sound = tfio.audio.AudioIOTensor(path)

			#print(sound)
			soundchunk = sound[0:soundlength]

			if(len(soundchunk) == soundlength):


				fsound = tf.math.reduce_sum(soundchunk, axis=[1]) #tf.squeeze(soundchunk, axis=[-1])
				soundlist.append(fsound)



	return tf.dtypes.cast(tf.stack(soundlist), tf.float32).numpy()

"""
Creates datax and datay:
	-datax is each raw sound file
	-datay is each spectrogram coming from the sound files
	
datax is converted into a spectrogram in-model
"""

def createData(soundpath):
	rawdatafiles = [soundpath + i for i in list(walk(soundpath))[0][2]]
	#print(rawdatapath)

	data = []
	for i in rawdatafiles:
		#print(i)
		if(i.split(".")[1] == "flac"):
			data.append(i)

	#print(data)

	lml = LogMelgramLayer(num_fft=NUM_FFT, hop_length=256)
	lml.build((1,8000))

	# mini = 0
	# maxi = 0
	tdata = []

	tldata = []

	for d in data:
		#print(d)
		sound = tfio.audio.AudioIOTensor(d)
		#print(sound)

		for i in range(len(sound)//soundlength):
			soundslice = sound[soundlength*i:soundlength*(i+1)]
			#print(soundslice)

			sqsound = tf.math.reduce_sum(soundslice, axis=[1])
			# print(sqsound)

			sqsound = tf.dtypes.cast(sqsound, tf.float32)
			#print(sqsound)

			#tlsound = tf.expand_dims(tf.math.reduce_sum(sqsound, axis=[-1]), axis=0)

			tdata.append(sqsound)
			# convert sound to spectrogram
			tldata.append(lml(tf.expand_dims(sqsound, axis=0)))

	datax = tf.dtypes.cast(tf.stack(tdata), tf.float32).numpy()
	datay = tf.dtypes.cast(tf.stack(tldata), tf.float32).numpy()

	return datax, datay

"""

Creates two models: 
model:
	-takes a goal sound as an input
		-the first layer converts it into a spectrogram
	-outputs a reconstruction of the goal sound
	-trains with the goal spectrogram as the x and y
predictor:
	-takes goal sound as input and converts it into a spectrogram
	-outputs the model's prediction for what coefficients are needed to create the goal sound

"""

def createModel(spectroInput):

	inp = Input(shape=(soundlength,))

	lml = LogMelgramLayer(num_fft=NUM_FFT, hop_length=256)

	x1 = Flatten()(lml(inp))
	xim1 = Dense(1000)(x1)
	xim2 = Dense(300)(xim1)
	x2 = Dense(numresourcesounds)(xim2)
	xf = numtospectrogramfunc(spectroInput)(numresourcesounds)(x2)

	model = keras.Model(inputs=inp, outputs=xf)
	predictor = keras.Model(inputs=inp, outputs=x2)

	return model, predictor

def predictSound(m, data, index):
	sqsound	= data[index]

	print(sqsound)
	print(sqsound.shape())

	return m.predict(sqsound)


if __name__ == "__main__":
	datax, datay = createData(soundpath)
	resourcesoundtensor = soundFilesToTensor(resourcesoundfiles)

	model, predictor = createModel(resourcesoundtensor)

	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='mean_squared_error', optimizer=opt)
	predictor.compile(loss='mean_squared_error', optimizer=opt)

	print(datax)
	print(datay)
	model.fit(x=datax, y=datay, batch_size=1)

	model.summary()

	print(predictSound(predictor, datax, 300))
