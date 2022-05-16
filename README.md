# Overlap_sound_gen
Desc: Given a goal sound to recreate and "resource sounds," a neural network learns to create coefficients to apply to the resource sounds to recreate the goal sound.

WIP - not currently functioning

overlappingsoundgen.py is the main file

Given Problem:
In a situation where real-time voice transmission is desired, computing power is available but bandwith is extremely limited, how can a deep neural network facilitate communication?
  
The Solution:
A deep neural network takes a desired sound as a input and outputs coefficients. These coefficients, when applied to the "resource sounds," recreate the original desired sound. In theory, these coefficients are what would be transmitted between parties.
  - resource sounds: a pool of sounds that both parties have access to
  - the deep neural network has the following configuration:
      - input: the "goal" sound
      - a layer that converts the goal sound into a spectrogram on the mel scale
      - hidden layers
      - two outputs, one for training and one for prediction:
          - prediction output: the coefficients that result in the goal sound once applied to the resource sounds. 
          - training output: a mel spectrogram generated by applying the coefficients to the resource sounds
          
  - x-data: the desired sound to recreate
  - y-data: a mel spectrogram of the desired sound


Notes:
  - a mel spectrogram is used because it represents sound on the mel scale--a scale that represents sound as humans hear it.
  - In my imagination, the use case for this project would be an astronaut far from Earth or a diver deep in the ocean.
