```
 _           _        _              _              
| |         | |      | |            (_)             
| |__  _ __ | |_ ___ | | _____ _ __  _ _______ _ __ 
| '_ \| '_ \| __/ _ \| |/ / _ \ '_ \| |_  / _ \ '__|
| |_) | |_) | || (_) |   <  __/ | | | |/ /  __/ |   
|_.__/| .__/ \__\___/|_|\_\___|_| |_|_/___\___|_|   
      | |                                           
      |_|       

                                  by Dave Randall
                                      MIT License
                                            2025

```
### A Byte Pair Encoding tokenizer in pure vanilla JS.

- Runs in the browser
- No dependencies

*Inspired by Andrej Karpathy's minibpe and tutorial.*
_____________________________

### Usage Examples

1. **Initialization**
```js
const vocabSize = 512;                                    // Set vocab size. (Default is 276)
const text = "Training text goes here.";                  // Provide training text.
const tokenizer = new BytePairTokenizer(text, vocabSize); // Initialize class.
tokenizer.train();                                        // Train the model.
```
2. **Encoding**
```js
const input = "Text to be encoded";                       // Text to be encoded
const encoded = tokenizer.encode(input);                  // Encode the text
// Pass encoded input to LLM / Save as embedding / etc.
```
3. **Decoding**
```js
const decoded = tokenizer.decode(encoded);                // Decode encoded text back to string
```
