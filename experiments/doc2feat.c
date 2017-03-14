// The word2vec part of this code has the following original copyright note:

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char corpus_file[MAX_STRING], tag_doc_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
// tomekd789...
char read_semantic_space_file[MAX_STRING];
char write_semantic_space_file[MAX_STRING];
char features_file[MAX_STRING], write_fs_file[MAX_STRING], read_fs_file[MAX_STRING];
int k_iter = 10, words_per_feat = 10;
// ...tomekd789
struct vocab_word *vocab;
int cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int wsd_repl = 0;
// tomekd789:
int sparse = 0; real sparse_threshold = 0.3;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, ssize = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, feats = 100; // tomekd789: changed from 0
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *sspace, *syn1, *syn1neg, *expTable, *fspace;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

struct word_similarity {
  long long cn;
  real sim;
};
struct word_similarity *feat_table;

// Original word2vec code
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries (original word2vec)
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word (original word2vec)
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// (original word2vec)
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary (original word2vec)
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary (original word2vec)
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts (original word2vec)
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens (original word2vec)
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes (original word2vec)
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

// Original word2vec
void LearnVocabFromCorpusFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(corpus_file, "rb");
  if (fin == NULL) {
    printf("ERROR: corpus file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 1000000 == 0)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in corpus file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

// Original word2vec
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

// Original word2vec
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in corpus file: %lld\n", train_words);
  }
  fin = fopen(corpus_file, "rb");
  if (fin == NULL) {
    printf("ERROR: corpus file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

// tomekd789
void SaveSemanticSpace() {
  FILE *fo = fopen(write_semantic_space_file, "wb");
  if (fo == NULL) {
    printf("ERROR: semantic space can't be written!\n");
    exit(1);
  }
  if (debug_mode > 1) {
    printf("%cSaving the semantic space                                           %c", 13, 13);
    fflush(stdout);
  }
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, ssize);
  long long i, j;
  for (i = 0; i < vocab_size; i++) {
    fprintf(fo, "%s ", vocab[i].word);
    for (j = 0; j < ssize; j++) fprintf(fo, "%lf ", sspace[i * ssize + j]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

// tomekd789
void ReadSemanticSpace() {
  FILE *fi = fopen(read_semantic_space_file, "rb");
  if (fi == NULL) {
    printf("ERROR: semantic space can't be read!\n");
    exit(1);
  }
  if (debug_mode > 1) {
    printf("%cReading the semantic space                                  %c", 13, 13);
    fflush(stdout);
  }
  long long v_size, l_size;
  fscanf(fi, "%lld %lld\n", &v_size, &l_size);
  if (v_size != vocab_size) {
    printf("ERROR: vocabulary size from the semantic space file does not match!\n");
    exit(1);
  }
  if (l_size != ssize) {
    printf("ERROR: semantic space size read from file does not match!\n");
    exit(1);
  }
  long long i, j;
  char word[MAX_STRING];
  for (i = 0; i < vocab_size; i++) {
    fscanf(fi, "%s ", word);
    if (strcmp(word, vocab[i].word) != 0) {
      printf("ERROR: vocabulary in the semantic space file does not match!\n");
      exit(1);
    }
    for (j = 0; j < ssize; j++) fscanf(fi, "%f ", &sspace[i * ssize + j]);
    fscanf(fi, "\n");
  }
  fclose(fi);
}

// tomekd789
void ReadFeatureSpace() {
  long long a, b;
  FILE *fi = fopen(read_fs_file, "rb");
  if (fi == NULL) {
    printf("ERROR: feature space can't be read!\n");
    exit(1);
  }
  if (debug_mode > 1) {
    printf("%cReading the feature space                                  %c", 13, 13);
    fflush(stdout);
  }
  long long ss_size;
  fscanf(fi, "%lld\n", &ss_size);
  if (ss_size != ssize) {
    printf("ERROR: feature space vector size from the feature space file does not match!\n");
    exit(1);
  }
  long long foo;
  for (a = 0; a < feats; a++) {
    if (feof(fi)) break;
    fscanf(fi, "%lld ", &foo);
    for (b = 0; b < ssize; b++) fscanf(fi, "%f ", &fspace[a * ssize + b]);
    fscanf(fi, "\n");
  }
  feats = a;
  fclose(fi);
}

// Original word2vec code
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&sspace, 128, (long long)vocab_size * ssize * sizeof(real));
  if (sspace == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * ssize * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < ssize; b++)
     syn1[a * ssize + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * ssize * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < ssize; b++)
     syn1neg[a * ssize + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < ssize; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    sspace[a * ssize + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / ssize;
  }
  CreateBinaryTree();
}

// Original word2vec code
void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(ssize, sizeof(real));
  real *neu1e = (real *)calloc(ssize, sizeof(real));
  FILE *fi = fopen(corpus_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < ssize; c++) neu1[c] = 0;
    for (c = 0; c < ssize; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < ssize; c++) neu1[c] += sspace[c + last_word * ssize];
        cw++;
      }
      if (cw) {
        for (c = 0; c < ssize; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * ssize;
          // Propagate hidden -> output
          for (c = 0; c < ssize; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < ssize; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < ssize; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * ssize;
          f = 0;
          for (c = 0; c < ssize; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < ssize; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < ssize; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < ssize; c++) sspace[c + last_word * ssize] += neu1e[c];
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * ssize;
        for (c = 0; c < ssize; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * ssize;
          // Propagate hidden -> output
          for (c = 0; c < ssize; c++) f += sspace[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < ssize; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < ssize; c++) syn1[c + l2] += g * sspace[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * ssize;
          f = 0;
          for (c = 0; c < ssize; c++) f += sspace[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < ssize; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < ssize; c++) syn1neg[c + l2] += g * sspace[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < ssize; c++) sspace[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

// Read or calculate vocab & semantic space; mostly original word2vec code
void GetModel() {
  long a;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromCorpusFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  InitNet();
  if (read_semantic_space_file[0] != 0) ReadSemanticSpace(); else {
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("Starting training using file %s\n", corpus_file);
    starting_alpha = alpha;
    if (negative > 0) InitUnigramTable();
    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  }
  if (write_semantic_space_file[0] != 0) SaveSemanticSpace();
}

// Calculate or read the feature space
void GetFeatureSpace() {
  fspace = (real *)calloc(feats * ssize, sizeof(real));
  if (read_fs_file[0] != 0) {
    ReadFeatureSpace();
    return;
  }
  // Run K-median on the word vectors, with the L1 norm
  // to generate K-median classes then taken as features
  int iter = k_iter, closeid;
  int *centcn = (int *)malloc(feats * sizeof(int));
  int *cl = (int *)calloc(vocab_size, sizeof(int));
  real closev, x;
  long long a, b, c, d;
  for (a = 0; a < vocab_size; a++) cl[a] = a % feats;
  // K-median loop
  for (a = 0; a < iter; a++) {
    for (b = 0; b < feats * ssize; b++) fspace[b] = 0;
    for (b = 0; b < feats; b++) centcn[b] = 1;
    for (c = 0; c < vocab_size; c++) {
      for (d = 0; d < ssize; d++) fspace[ssize * cl[c] + d] += sspace[c * ssize + d];
      centcn[cl[c]]++;
    }
    // Normalize length vs count; it's then NOT normalized wrt L2 norm
    for (b = 0; b < feats; b++)
      for (c = 0; c < ssize; c++) fspace[ssize * b + c] /= centcn[b];
    for (c = 0; c < vocab_size; c++) {
      if ((debug_mode > 1) && (c % 1000 == 0)) {
        printf("%cK-median iteration %lld of %d; vocab word %lldK of %lldK            %c",
                13, a+1, iter, c/1000, vocab_size/1000, 13);
        fflush(stdout);
      }
      closev = 1e10; // +\infty
      closeid = 0;
      for (d = 0; d < feats; d++) {
        x = 0;
        real y = 0;
        for (b = 0; b < ssize; b++) {
          y = fspace[ssize * d + b] - sspace[c * ssize + b]; // L1 norm
          if (y < 0) y = -y;
          x += y;
        }
        if (x < closev) {
          closev = x;
          closeid = d;
        }
      }
      cl[c] = closeid;
    }
  }
  free(centcn); free(cl);
  // Normalize classes vectors, L2 norm, for subsequent cosine similarity calculations
  for(a = 0; a < feats; a++) {
    real len_class = 0;
    for (b = 0; b < ssize; b++) len_class += fspace[a * ssize + b] * fspace[a * ssize + b];
    len_class = sqrt(len_class);
    if (len_class == 0) {
      printf("ERROR: a class vector with zero length.\n");
      printf("This should never happen under regular circumstances, please report it as a bug ");
      printf("( tomekd789@gmail.com )\n");
      exit(1);
    }
    for (b = 0; b < ssize; b++) fspace[a * ssize + b] /= len_class;
  }

  // Remove duplicates from the feature space (i.e. vectors too close to one another)
  a = 0;
  while (1) {
    if (a >= feats - 1) break; // The main loop has reached the last element; nothing to compare
    b = a + 1;
    while (1) {
      if (b >= feats) break;
      real x = 0;
      // calculate the cosine similarity between the two class vectors (a-th to b-th)
      for (c = 0; c < ssize; c++) x += fspace[ssize * a + c] * fspace[ssize * b + c];
      if (x > 0.9) { // Features are too close to each other
        for (c = b; c < feats - 1; c++)
          for (d = 0; d < ssize; d++)
            fspace[ssize * c + d] = fspace[ssize * (c+1) + d];
        feats--;
      }
      b++;
    }
    a++;
  }

  // Write feature space to fs
  if (write_fs_file[0] != 0) {
    FILE *fs = fopen(write_fs_file, "wb");
    if (fs == NULL) {
      printf("ERROR: cannot write feature space to file!\n");
      exit(1);
    }
    if (debug_mode > 1) {
      printf("%cSaving the feature space                                            %c", 13, 13);
      fflush(stdout);
    }
    fprintf(fs, "%lld\n", ssize);
    for (a = 0; a < feats; a++) {
      fprintf(fs, "%lld ", a);
      for (b = 0; b < ssize; b++) fprintf(fs, "%lf ", fspace[a * ssize + b]);
      fprintf(fs, "\n");
    }
    fclose(fs);
  }
}

// tomekd789
// Generate human readable features
void GenerateFeatures() {
  long long a, b, c, d;
  feat_table = (struct word_similarity *)malloc(feats * words_per_feat * sizeof(struct word_similarity));
  for (a = 0; a < feats; a++)
    for (b = 0; b < words_per_feat; b++) {
      feat_table[a * words_per_feat + b].cn = 0; // initialize feat_table word indexes with 0
      feat_table[a * words_per_feat + b].sim = -10; // and similarities with -\infty wrt cosine similarity
    }
  for (a = 0; a < vocab_size; a++) {
    if ((debug_mode > 1) && (a % 1000 == 0)) {
      printf("%cFeatures generation: %lldK of %lldK words processed            %c", 13, a/1000, vocab_size/1000, 13);
      fflush(stdout);
    }
    for (b = 0; b < feats; b++) {
      real x = 0;
      // calculate the cosine similarity between the class vector and the word vector
      real len = 0;
      for (c = 0; c < ssize; c++) {
        x += fspace[ssize * b + c] * sspace[a * ssize + c];
        len += sspace[a * ssize + c] * sspace[a * ssize + c];
      }
      x /= sqrt(len); // sspace might be pre-normalized, but decided not to touch global tables; too large to copy
      // insert the word&sim to its sorted position in the feat_table
      for (d = words_per_feat - 1; (d >= 0) && (x > feat_table[b * words_per_feat + d].sim); d--);
      d++;
      // now x <= t[b, d-1].sim, and x > t[b, d].sim,
      // assuming t[b, -1].sim = -\infty, and t[b, words_per_feat] = \infty
      // d \in (0 .. words_per_feat)
      // hence d is the insertion point
      long dd;
      for (dd = words_per_feat - 1; dd > d; dd--) { // shifting elements greater than x
        feat_table[b * words_per_feat + dd].cn   = feat_table[b * words_per_feat + dd-1].cn;
        feat_table[b * words_per_feat + dd].sim = feat_table[b * words_per_feat + dd-1].sim;
      }
      if (d < words_per_feat) {
        feat_table[b * words_per_feat + d].cn = a;
        feat_table[b * words_per_feat + d].sim = x;
      }
    }
  }

  // Write feat_table to ft
  if (features_file[0] != 0) {
    FILE *ft = fopen(features_file, "wb");
    if (ft == NULL) {
      printf("ERROR: cannot write features to file!\n");
      exit(1);
    }
    for (b = 0; b < feats; b++) {
      fprintf(ft, "\nFeature %lld\n", b);
      for (d = 0; d < words_per_feat; d++)
        fprintf(ft, "%s %f\n",
                vocab[feat_table[b * words_per_feat + d].cn].word,
                feat_table[b * words_per_feat + d].sim);
    }
    fclose(ft);
  }
}

// tomekd789
void GenerateTags() {
  long long a, b;
  if (tag_doc_file[0] != 0) {
    real *acc_vec = (real *)malloc(ssize * sizeof(real));
    FILE *fc = fopen(corpus_file, "rb");
    FILE *ft = fopen(tag_doc_file, "wb");
    if (ft == NULL) {
      printf("ERROR: cannot write topics to file!\n");
      exit(1);
    }
    // Current document processed
    long long doc_processed = 0;
    // Total number of documents
    long long max_documents = vocab[0].cn;
    // For reading tokens from the input file
    char word[MAX_STRING];
    // To count features with at least document closer than 0.5, or farther than -0.5
    typedef int bool;
    #define TRUE 1
    #define FALSE 0
    bool *significant_features = (bool *)malloc(feats * sizeof(bool));
    for (a = 0; a < feats; a++) significant_features[a] = FALSE;
    int sig_feat_cnt = 0;
    // Loop over documents in the input stream
    while(1) {
      for(a = 0; a < ssize; a++) acc_vec[a] = 0;
      // Loop over words in the current input document
      while(1) {
        ReadWord(word, fc);
        // If end-of-doc or end-of-stream quit the words loop
        if ((feof(fc)) || (strcmp (word, (char *)"</s>") == 0)) break;
        long long vocab_index = SearchVocab(word);
        //  If found add its vector to the accumulator
        if (vocab_index > -1) for(a = 0; a < ssize; a++) acc_vec[a] += sspace[vocab_index * ssize + a];
        // This is inefficient, and could be replaced with counting words and multiplying vectors by them.
        // However it does not impact the total runtime significantly, then it is left as-is for simplicity.
      }
      if (sparse) fprintf(ft, "Document %lld\n", doc_processed);
      // Normalize accumulator for the cosine similarity calculation
      real len_acc = 0;
      for (a = 0; a < ssize; a++) len_acc += acc_vec[a] * acc_vec[a];
      len_acc = sqrt(len_acc);
      if (len_acc > 0) for (a = 0; a < ssize; a++) acc_vec[a] /= len_acc; // NULL document will give len_acc == 0
      // For each class, i.e. feature
      for(a = 0; a < feats; a++) {
        // Calculate the cosine similarity between the accumulator and the i-th class vector
        real x = 0;
        for (b = 0; b < ssize; b++) x += acc_vec[b] * fspace[a * ssize + b];
        // Write the feature num and the cosine similarity calculated
        if (sparse) {
          if ((x > 0.3) || (x < -0.3)) fprintf(ft, " %lld %f", a, x);
        } else {
          fprintf(ft, "%f", x);
          if (a < feats-1) fprintf(ft, ";");
        }
        if ((x < -0.5) || (x > 0.5)) significant_features[a] = TRUE;
      }
      // Display progress information
      if ((debug_mode > 1) && (doc_processed % 1000 == 0)){
        printf("%c%lldK of %lldK documents tagged                      %c", 13, doc_processed/1000, max_documents/1000, 13);
        fflush(stdout);
      }
      fprintf(ft, "\n");
      if (feof(fc)) break;
      doc_processed++; //if (doc_processed == 100) break;
    }
    for (a = 0; a < feats; a++) if (significant_features[a]) sig_feat_cnt++;
    if (debug_mode > 0) printf("%c%d significant features (> 0.5) found.                     \n", 13, sig_feat_cnt);
    fclose(fc); fclose(ft);
  }
}

void WsdRepl() {
  int a, b, c, d, cn;
  real x;
  real *vec = (real *)malloc(ssize * sizeof(real));
  if (wsd_repl) {
    char input_phrase[MAX_STRING * 10];
    char phrase_words[10][MAX_STRING]; // up to 10 words in the input phrase
    while (1) {
      printf("Enter word or phrase (EXIT to break): ");
      d = 0;
      while (1) { // read input phrase
        input_phrase[d] = fgetc(stdin);
        if ((input_phrase[d] == '\n') || (d >= MAX_STRING - 1)) {
          input_phrase[d] = 0;
          break;
        }
        d++;
      }
      if (!strcmp(input_phrase, "EXIT")) break;
      b = 0; c = 0; cn = 0;
      while (1) {  // split input_phrase to words
        phrase_words[cn][b] = input_phrase[c];
        b++; c++;
        phrase_words[cn][b] = 0;
        if (input_phrase[c] == 0) break;
        if (input_phrase[c] == ' ') {cn++; c++; b = 0;}
      }
      cn++;
      for (a = 0; a < ssize; a++) vec[a] = 0;
      for (a = 0; a < cn; a++) { // loop over phrase words
        char word[MAX_STRING];
        for (b = 0; phrase_words[a][b] != 0; b++) word[b] = phrase_words[a][b];
        word[b] = 0;
        long long vocab_index = SearchVocab(word);
        if (vocab_index == -1) {
          printf("Warning: %s not found in the vocabulary.\n", word);
        } else {
          //  If found add its vector to the accumulator
          for(b = 0; b < ssize; b++) vec[b] += sspace[vocab_index * ssize + b];
        }
      }
      // vec L_2 normalization
      real len = 0;
      for (a = 0; a < ssize; a++) len += vec[a] * vec[a];
      len = sqrt(len);
      for (a = 0; a < ssize; a++) vec[a] /= len;
      // Print features close to the vector
      for (a = 0; a < feats; a++) {
        x = 0;
        for (b = 0; b < ssize; b++) x += vec[b] * fspace[ssize * feats + b];
        if (x > 0.3) {
          printf("%lld: ", a);
          for (d = 0; d < words_per_feat; d++)
            printf("%s ", vocab[feat_table[b * words_per_feat + d].cn].word);
          printf("\n");
        }
      }
    }
  }
  free(vec); free(fspace);
}

// Original word2vec code
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

// Mostly original word2vec code, with obvious subsequent changes
int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-corpus <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-tag-doc <file>\n");
    printf("\t\tUse <file> to save tagged documents\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t-k-iter <int>\n");
    printf("\t\tRun <int> K-median iterations (default 10)\n");
    printf("\t-words-per-feat <int>\n");
    printf("\t\tWrite <int> words for each feature (default 10)\n");
    printf("\t-sparse <int>\n");
    printf("\t\tProvide sparse output, cut off at 0.3; default is 0 (not used)\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-feats <int>\n");
    printf("\t\tInitial number of features, i.e. K-median classes; default is equal to -size\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-read-semantic-space <file>\n");
    printf("\t\tThe semantic space will be read from <file>, not computed by word2vec\n");
    printf("\t-save-semantic-space <file>\n");
    printf("\tThe semantic space will be saved to <file>\n");
    printf("\t-features <file>\n");
    printf("\tFeatures will be saved to <file> (proxy word sets)\n");
    printf("\t-save-f-space <file>\n");
    printf("\tThe feature space will be saved to <file> (vectors)\n");
    printf("\t-vocab-max-size <int>\n");
    printf("\tThe maximum vocabulary size will be vocab-max-size * 0.7 words; default is 30M\n");
    printf("\t-wsd-repl <int>\n");
    printf("\tRun WSD REPL (default is no)\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./doc2feat -corpus data.txt -tag-doc tags.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  tag_doc_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  read_semantic_space_file[0] = 0;
  write_semantic_space_file[0] = 0;
  features_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) ssize = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-corpus", argc, argv)) > 0) strcpy(corpus_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-semantic-space", argc, argv)) > 0) strcpy(read_semantic_space_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-semantic-space", argc, argv)) > 0) strcpy(write_semantic_space_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-features", argc, argv)) > 0) strcpy(features_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-f-space", argc, argv)) > 0) strcpy(write_fs_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-tag-doc", argc, argv)) > 0) strcpy(tag_doc_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-k-iter", argc, argv)) > 0) k_iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-words-per-feat", argc, argv)) > 0) words_per_feat = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sparse", argc, argv)) > 0) sparse = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-wsd-repl", argc, argv)) > 0) wsd_repl = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-vocab-max-size", argc, argv)) > 0) vocab_max_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-feats", argc, argv)) > 0) {feats = atoi(argv[i + 1]);} else feats = ssize;
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  GetModel();
  GetFeatureSpace();
  GenerateFeatures();
  if (tag_doc_file[0] != 0) GenerateTags();
  if (wsd_repl) WsdRepl();
  return 0;
}
