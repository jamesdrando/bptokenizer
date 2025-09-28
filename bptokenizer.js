/**
 *  _           _        _              _              
 * | |         | |      | |            (_)             
 * | |__  _ __ | |_ ___ | | _____ _ __  _ _______ _ __ 
 * | '_ \| '_ \| __/ _ \| |/ / _ \ '_ \| |_  / _ \ '__|
 * | |_) | |_) | || (_) |   <  __/ | | | |/ /  __/ |   
 * |_.__/| .__/ \__\___/|_|\_\___|_| |_|_/___\___|_|   
 *       | |                                           
 *       |_|       
 * 
 *                                     Dave Randall
 *                                      MIT License
 *                                             2025
*/

/**
 * ### A Byte Pair Encoding tokenizer in pure vanilla JS.
 * 
 * *Inspired by Andrej Karpathy's minibpe and tutorial.*
 * 
 * - Runs in the browser
 * - No dependencies
 * 
 * _____________________________
 * 
 * ### Usage Examples
 * 
 * #############################
 * 
 * #### => Initialization =>
 * 
 * **const** vocabSize = 512; // *Default is 276*
 * 
 * **const** text = "Training text goes here."
 * 
 * **const** tokenizer = **new** BytePairTokenizer(text, vocabSize);
 * 
 * #### => Training =>
 * 
 * tokenizer.train(); // *Primes the tokenizer.*
 * 
 * #### => Encoding =>
 * 
 * **const** input = "Text to be encoded";
 * 
 * **const** encoded = tokenizer.encode(input);
 * 
 *  **[Pass encoded input to LLM / Save as embedding / etc.]**
 * 
 * #### => Decoding =>
 * 
 * **const** decoded = tokenizer.decode(encoded);
 */

class BytePairTokenizer {
  /**
   * Create an instance of the BytePairTokenizer.
   * Must be trained before use by calling the
   * .train() method.
   * 
   * @param {String} text The training text.
   * @param {Number} vocabSize The size of the tokenizer vocab.
   */
  constructor(text, vocabSize=276) {
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
    this.tokens = this.encoder.encode(text);
    this.vocabSize = vocabSize;
  }

  // ------------------------------------------------------------

  /**
   * Trains the tokenizer using the provided text.
   */
  train() {
    // To reach our target vocabSize, we n merges
    // n = (Target Vocab Size) - (Base Vocab Size)
    // Text Encoder returns a byte array (Uint8)
    // This means our base vocabSize is 256
    const nMerges = this.vocabSize - 256;

    // Copy so we don't overwrite original tokens
    this.ids = new Uint16Array(this.tokens);
    
    // We need to store the merges for encoding / decoding
    this.merges = new Map();

    // Store the inverse for convenience later.
    this.idxmap = new Map();

    for (let i = 0; i < nMerges; i++) {
      let stats = BytePairTokenizer.getStats(this.ids); 
      let pair = BytePairTokenizer.getMax(stats); 
      let idx = 256 + i; // This keeps idx outside of our base range (0 - 255)
      this.ids = BytePairTokenizer.merge(this.ids, pair, idx);

      // Store both both ways for encoding / decoding
      this.merges.set(pair, idx);
      this.idxmap.set(idx, pair);
    }
    this.trained = true; 
  }

  // ------------------------------------------------------------
  
  /**
   * Encodes a string using the trained tokenizer.
   * 
   * @param {String} inputText Any arbitrary string input.
   * @returns 
   */
  encode(inputText) {
    // If the model is untrained, throw error
    if (!this.trained) {
      throw Error("BytePairTokenizer is untrained.")
    }

    // Convert input text to bytes (Uint8Array)
    const inputTokens = this.encoder.encode(inputText);
    let ids = new Uint16Array(inputTokens);

    for (const merge of this.merges) {
      let [pair, idx] = merge
      ids = BytePairTokenizer.merge(ids, pair, idx);
    }
    return ids;
  }

  // ------------------------------------------------------------

  /**
   * Decodes endcoded tokens.
   * 
   * @param {Uint16Array} outputTokens The encoded output to be decoded.
   * @returns {String}
   */

  decode(outputTokens) {
    // If the model is untrained, throw error
    if (!this.trained) {
      throw Error("BytePairTokenizer is untrained.")
    }

    let tokens = [];
    for (let i = 0; i < outputTokens.length; i++) {
      const token = outputTokens[i];
      
      if(token < 256) {
        tokens.push(token);
    
      } else {
        let pair = this.idxmap.get(token).split(',').map((id) => parseInt(id));        

        // Recursively decode pairs until all are valid 8 bit numbers
        const left = this.decode(new Uint16Array([pair[0]]));
        const right = this.decode(new Uint16Array([pair[1]]));
        
        tokens.push(...this.encoder.encode(left + right));
      }
    }    
    return this.decoder.decode(new Uint8Array(tokens)); 
  }

  // ------------------------------------------------------------

  /**
   * Returns the most frequent (or equally most frequent) Byte Pair.
   * 
   * @param {Map<String, Number>} stats A map of Byte Pair and frequency.
   * @returns 
   */
  static getMax(stats) {
    let max = 0;
    let maxPair = "";
    
    for (const [pair, count] of stats) {
      if (count > max) {
        max = count;
        maxPair = pair;
      }
    }
    return maxPair;
  }

  // ------------------------------------------------------------

  /**
   * Returns a Map of <Byte Pair, frequency>. For use in identifying
   * the most frequent byte pairs.
   *
   * @param {Array<Number>} ids Array of tokens
   * @returns {Map<String, Number>}
   * @static
   */
  static getStats(ids) {
    if (ids.length < 2) {
      throw Error(`Cannot evaluate pairs with less that 2 values. \n ids: ${ids}`);
    }
    const stats = new Map();

    for (let i = 0; i < ids.length - 1; i++) {
      const pair = tuple(ids[i], ids[i+1]);
      stats.set(pair, (stats.get(pair) || 0) + 1);
    }
    return stats;
  }

  // ------------------------------------------------------------

  /**
   * Merges all occurences of the given byte pair with idx and
   * returns the updated ids Array.
   * 
   * @param {Array<Number>} ids Array of tokens.
   * @param {String} pair Target Byte Pair, represented as a String: `${u8_1},{u8_2}`
   * @param {Number} idx The id that will replace the Byte Pair.
   * @returns {Array<Number>}
   */

  static merge(ids, pair, idx) {
    const newIds = [];
    let i = 0;

    // We need to check i,i+1
    // Let's not check bounds every iteration (slower)
    // Instead, lets check length - 1
    // Then, we can get the last token at the end
    while (i < ids.length - 1) {
      // If the pair is not our target, do not merge
      if (!(tuple(ids[i], ids[i+1]) == pair)) {
        newIds.push(ids[i]);
        i += 1;
      } else {
        newIds.push(idx);
        i += 2;
      }
    }
    
    // Check i so we don't miss any tokens
    if (i < ids.length) {
      newIds.push(ids[i]);
    }

    return newIds;
  }
}

/**
 * For lack of a collection-type with consistent equality checks,
 * we resort to string tuples. 
 * @param  {...Number} args Numbers that will be "," joined.
 * @returns 
 */
function tuple(...args) {
  return args.join(",");
}

module.exports = {BytePairTokenizer, tuple}
