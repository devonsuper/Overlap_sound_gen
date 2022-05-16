import pydub
from os import walk

path = 'C:/Users/devon/Desktop/Programming Programs/Machine Learning and AI/overlapcreation/ussounds/'
paths = list(walk(path))[0][2]

for i in paths:
	if(".wav" in i):
		song = pydub.AudioSegment.from_wav(path + i)
		print(path +i.split(".")[0] + ".flac")
		song.export(path + i.split(".")[0] + ".flac", format = "flac")