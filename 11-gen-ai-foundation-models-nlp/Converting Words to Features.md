one-hot encoding
- method used to convert categorical data into feature vectors that a neural network can understand

## Big picture (first principles)

Neural networks only understand **numbers**.  
Text → tokens → numbers → vectors → neural network.

Everything here answers **one question**:

> _How should we turn words into numbers so a model can learn something useful?_

---

## 1️⃣ One-hot encoding — the simplest idea

**Idea:**  
Each word is just an ID. Represent it as a vector with:

- length = vocabulary size
- exactly **one 1**, everything else **0**

Example:

```
Vocabulary: ["I", "like", "cats", "dogs"]
"cats" → [0, 0, 1, 0]
```

**Why it works**

- No ambiguity
- Simple
- Neural nets can consume it

**Why it breaks**

- Huge vectors (vocab = 50k → 50k dimensions)
- No notion of similarity  
    ("cats" and "dogs" are as unrelated as "cats" and "quantum")

**Check:**  
If two words have different one-hot vectors, does the model know they’re related?  
(Answer yes/no — don’t overthink.) No

---

## 2️⃣ Bag of Words — from words to documents

One-hot encodes **words**.  
Bag of Words (BoW) encodes **documents**.

**How**

- Add (or average) the one-hot vectors of all words in a document
- Order is ignored

Example:

```
"I like cats"
= one-hot("I") + one-hot("like") + one-hot("cats")
```

**What BoW captures**

- Which words appear
- How often they appear

**What it loses**

- Word order
- Context ("not good" vs "good")

**Key insight from the lesson**

> A BoW vector is just a big sparse vector fed into a linear layer

**Check:**  
If I shuffle the words in a sentence, does BoW change? No

---

## 3️⃣ Embeddings — learning meaning instead of IDs

This is the **conceptual leap**.

Instead of:

```
word → huge sparse vector
```

We learn:

```
word → small dense vector (embedding)
```

Example:

```
"cats" → [ 0.12, -0.87, 0.44, ... ]  (say, 50 dims)
```

**What changed**

- Vectors are **learned**, not fixed
- Similar words → similar vectors
- Dimensionality drops dramatically

**Critical mental model**  
An **embedding layer is just a lookup table**:

```
token index → row in an embedding matrix
```

That’s why the lesson says:

> Using a token index is equivalent to multiplying a one-hot vector by the embedding matrix

**Check:**  
Why doesn’t the model need the one-hot vector anymore once embeddings exist? Because the embeddings capture nuance and relationship between words rather than just if a word exists or not.

---

## 4️⃣ Embedding Bag — BoW, but efficient and modern

This part often clicks last.

Recall:

- BoW + linear layer = sum of word embeddings
- But computing BoW explicitly is wasteful

**EmbeddingBag does this directly**

```
[input: token indices]
→ lookup embeddings
→ sum or average them
```

No giant sparse vectors. No explicit BoW.

**Why it matters**

- Faster
    
- Less memory
    
- Standard in real NLP systems
    

From the lesson:

> Feeding a bag-of-words vector into a hidden layer is mathematically the same as summing embeddings

**Check:**  
What information does EmbeddingBag _still_ ignore that transformers care deeply about?

---

## 5️⃣ How this fits into a neural network (end-to-end)

Typical pipeline you just learned:

```
Text
 → Tokenize
 → Token IDs
 → Embedding / EmbeddingBag
 → Neural layers
 → Output (classification)
```

**Key unifying idea**

> Embeddings replace one-hot + linear layers with something learnable, compact, and semantic.

---

## One-sentence summaries (memory anchors)

- **One-hot**: identity, no meaning
    
- **Bag of Words**: counts, no order
    
- **Embedding**: meaning in vectors
    
- **EmbeddingBag**: BoW done right
    

---

## Optional next step (your choice)

We can do **one** of these next:

1. Walk through a tiny PyTorch example line by line
    
2. Compare EmbeddingBag vs Transformer embeddings
    
3. Do a “teach it back” round where _you_ explain one concept and I refine it
    

Tell me which path you want 👇