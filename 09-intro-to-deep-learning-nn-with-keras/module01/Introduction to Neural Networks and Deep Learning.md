## Deep Learning

Applications of Deep Learning

- Color Restoration - automatic colorization and color restoration in black and white images - https://www.theverge.com/2016/4/27/11519272/these-old-black-and-white-photos-were-colorized-by-artificial
	- http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/extra.html (404)
- Speech Reenactment
	- syncing lip movements in a video with an audio clip
	- University of Washington first to build realistic results
	- audio2video - synthesizing video from video data and lip sync audio
	- video2video - extract audio from one video and lip sync audio with another
- Automatic Handwriting Generation
	- Alex Graves - University of Toronto - RNN
- Other Applications
	- Automatic Machine Translation
	- Adding Sounds to Silent Movies
	- Object Classification and Detection in Images
	- Self-Driving Cars

___
## Neurons and Neural Networks

https://www.khanacademy.org/science/biology/human-biology/neuron-nervous-system/a/overview-of-neuron-structure-and-function

![[videoframe_108990.png]]

![[3567fc3560de474001ec0dafb068170d30b0c751.png]]

## Artificial Neural Networks

3 main topics associate with artificial neural networks
- forward propagation
- backpropagation
- activation functions

### Forward Propagation
the process through which data passes through layers of neurons in a neural network from the input layer all the way to the output layer

input layer -> hidden layers -> output player

![[videoframe_99525.png]]

- every connection has a specific weight (w) by which the flow of data is regulated; x1 and x2 are the two inputs (integer or float)
- when the input passes through its connection it is adjusted depending on the connection weight
- neuron processes the inputs, weights, and a constant known as the bias
- z -> linear combination of the inputs, weights, and bias
- a => z -> a being the output of the network
### Backpropagation


### Activation Functions

![[videoframe_203241.png]]

- non-linear transformations like the sigmoid function are called activation functions
- decide whether a neuron should be activated or not
- a neural network without an activation function is essentially just a linear regression model