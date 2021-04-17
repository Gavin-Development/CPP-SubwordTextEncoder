# SubwordTextEncoder Library
C++ implemented Subword Text Encoder for Natural Language Processing.

## Motivations & Inspirations
- [SubwordTextEncoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder)
- Originally Created for speeding up the [Gavin Project.](https://github.com/Scot-Survivor/GavinTraining)
- I wanted to properly test out C++, figured re-creating something like this is the way to go.

## So how does it work?
Well, the reason I use a Subword Text Encoder(STE), over a regular Encoder is because 
STEs are able to encode words they've never seen before. Through Byte encoding. 
If the STE is given a word not in its vocabulary then it splits the word into letters 
and encodes each individual letter. Therefore while models using the STE cannot come up with
these words alone, they can understand them and link them to other ones.

## How to use this
The STE, is a simple class inside the namespace "tokenizers" (this is so I can add more
tokenizers in the future if I feel the need.), The actual STE class is called 
"SubwordTextEncoder". 
### Constructors
To create an instance there are 2 ways. 
- `tokenizers::SubwordTextEncoder(target_vocab_size, name)`
or
- `tokenizer::SubwordTextEncoder(filename_prefix)`

For the first Constructor, you pass 2 variables, first one decides how large the vocabulary
size should be, while the name is just a way to keep track of different STEs. 
The second method, allows you to load a previously saved STE. 
### Generating a vocabulary
Assuming you haven't loaded a previous STE, you can choose to generate a vocabulary
through its method `SubwordTextEncoder.build_vocabulary(texts)` this methods
takes a list of strings, each one a sentence. From this it gets the most common, 
words and add those to the vocabulary. This is the biggest difference from TF's 
implementation and also the major draw back. Note: This is a mutli-threaded method,
it will try to use as much of your CPU as it can. 
### Encoding & Decoding
Whether you have loaded, or built your own vocabulary the process for encoding & decoding
is the same. To encode a string just pass it to the encode method
`SubwordTextEncoder.encode(string)`, this will return a list of integer ID's. 
To decode a list of ID's merely pass it into the decode method. 
`SubwordTextEncoder.decode(string)` NOTE: All ID's are +1 its value inside the vocabulary 
to allow for the "0" ID to be used as padding inside a model. 