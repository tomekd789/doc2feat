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

// tomekd789: any enhancements added further are clearly annotated as such
// in comments containing 'tomekd789'.

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

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
// tomekd789...
char read_semantic_space_file[MAX_STRING];
char write_semantic_space_file[MAX_STRING];
char features_file[MAX_STRING], fs_file[MAX_STRING];
int k_iter = 10, words_per_feat = 10;
// ...tomekd789
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
// tomekd789:
int sparse = 0; real sparse_threshold = 0.3;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 100; // tomekd789: changed from 0
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

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

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
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

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
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

// Sorts the vocabulary by frequency using word counts
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

// Reduces the vocabulary by removing infrequent tokens
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
// Frequent words will have short uniqe binary codes
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

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
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
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

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
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
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
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  long long i, j;
  for (i = 0; i < vocab_size; i++) {
    fprintf(fo, "%s ", vocab[i].word);
    if (binary) for (j = 0; j < layer1_size; j++) fwrite(&syn0[i * layer1_size + j], sizeof(real), 1, fo);
    else for (j = 0; j < layer1_size; j++) fprintf(fo, "%lf ", syn0[i * layer1_size + j]);
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
  if (l_size != layer1_size) {
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
    for (j = 0; j < layer1_size; j++) fscanf(fi, "%f ", &syn0[i * layer1_size + j]);
    fscanf(fi, "\n");
  }
  fclose(fi);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
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
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
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
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
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
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
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
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
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

void TrainModel() {
  long a; // tomekd789: , b, c, d;
  // tomekd789: commented out
  //FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  // tomekd789
  if (read_semantic_space_file[0] != 0) ReadSemanticSpace(); else {
    if (negative > 0) InitUnigramTable();
    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  }
  // tomekd789
  if (write_semantic_space_file[0] != 0) SaveSemanticSpace();
  // tomekd789: the rest of the original code has been moved to CalculateTopics()
}

// tomekd789
void CalculateTopics() {

  // Run K-means on the word vectors, with the L1 norm
  // to generate K-means classes then taken as features
  int clcn = classes, iter = k_iter, closeid;
  int *centcn = (int *)malloc(clcn * sizeof(int));
  int *cl = (int *)calloc(vocab_size, sizeof(int));
  real closev, x;
  real *fspace = (real *)calloc(clcn * layer1_size, sizeof(real));
  long long a, b, c, d;
  for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
  // K-means loop
  for (a = 0; a < iter; a++) {
    for (b = 0; b < clcn * layer1_size; b++) fspace[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < vocab_size; c++) {
      for (d = 0; d < layer1_size; d++) fspace[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
      centcn[cl[c]]++;
    }
    // Normalize length vs count; it's then NOT normalized wrt L2 norm
    for (b = 0; b < clcn; b++)
      for (c = 0; c < layer1_size; c++) fspace[layer1_size * b + c] /= centcn[b];
    for (c = 0; c < vocab_size; c++) {
      if ((debug_mode > 1) && (c % 1000 == 0)) {
        printf("%cK-means iteration %lld of %d; vocab word %lldK of %lldK            %c",
                13, a+1, iter, c/1000, vocab_size/1000, 13);
        fflush(stdout);
      }
      closev = 1e10; // +\infty
      closeid = 0;
      for (d = 0; d < clcn; d++) {
        x = 0;
        real y = 0;
        for (b = 0; b < layer1_size; b++) {
          y = fspace[layer1_size * d + b] - syn0[c * layer1_size + b]; // L1 norm
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
  free(centcn);
  // Normalize classes vectors, L2 norm, for subsequent cosine distance calculations
  for(a = 0; a < clcn; a++) {
    real len_class = 0;
    for (b = 0; b < layer1_size; b++) len_class += fspace[a * layer1_size + b] * fspace[a * layer1_size + b];
    len_class = sqrt(len_class);
    if (len_class == 0) {
      printf("ERROR: a class vector with zero length.\n");
      printf("This should never happen under regular circumstances, please report it as a bug ");
      printf("( tomekd789@gmail.com )\n");
      exit(1);
    }
    for (b = 0; b < layer1_size; b++) fspace[a * layer1_size + b] /= len_class;
  }

  // Generate and save features
  FILE *ft = fopen(features_file, "wb");
  if (ft == NULL) {
    printf("ERROR: cannot write features to file!\n");
    exit(1);
  }
  struct word_distance {
    long long cn;
    real dist;
  };
  struct word_distance *feat_table =
    (struct word_distance *)malloc(clcn * words_per_feat * sizeof(struct word_distance));
  for (a = 0; a < clcn; a++)
    for (b = 0; b < words_per_feat; b++) {
      feat_table[a * words_per_feat + b].cn = 0; // initialize feat_table word indexes with 0
      feat_table[a * words_per_feat + b].dist = -10; // and distances with -\infty wrt cosine distance
    }
  for (a = 0; a < vocab_size; a++) {
    if ((debug_mode > 1) && (a % 1000 == 0)) {
      printf("%cFeatures generation: %lldK of %lldK words processed            %c", 13, a/1000, vocab_size/1000, 13);
      fflush(stdout);
    }
    for (b = 0; b < clcn; b++) {
      real x = 0;
      // calculate the cosine distance between the class vector and the word vector
      real len = 0;
      for (c = 0; c < layer1_size; c++) {
        x += fspace[layer1_size * b + c] * syn0[a * layer1_size + c];
        len += syn0[a * layer1_size + c] * syn0[a * layer1_size + c];
      }
      x /= sqrt(len); // syn0 might be pre-normalized, but decided not to touch global tables; too large to copy
      // insert the word&dist to its sorted position in the feat_table
      for (d = words_per_feat - 1; (d >= 0) && (x > feat_table[b * words_per_feat + d].dist); d--);
      d++;
      // now x <= t[b, d-1].dist, and x > t[b, d].dist,
      // assuming t[b, -1].dist = -\infty, and t[b, words_per_feat] = \infty
      // d \in (0 .. words_per_feat)
      // hence d is the insertion point
      long dd;
      for (dd = words_per_feat - 1; dd > d; dd--) { // shifting elements greater than x
        feat_table[b * words_per_feat + dd].cn   = feat_table[b * words_per_feat + dd-1].cn;
        feat_table[b * words_per_feat + dd].dist = feat_table[b * words_per_feat + dd-1].dist;
      }
      if (d < words_per_feat) {
        feat_table[b * words_per_feat + d].cn = a;
        feat_table[b * words_per_feat + d].dist = x;
      }
    }
  }

  // Write feat_table to ft
  for (b = 0; b < clcn; b++) {
    fprintf(ft, "\nFeature %lld\n", b);
    for (d = 0; d < words_per_feat; d++)
      fprintf(ft, "%s %f\n",
              vocab[feat_table[b * words_per_feat + d].cn].word,
              feat_table[b * words_per_feat + d].dist);
  }

  // Write feature space to fs
  if (fs_file[0] != 0) {
    FILE *fs = fopen(fs_file, "wb");
    if (ft == NULL) {
      printf("ERROR: cannot write feature space to file!\n");
      exit(1);
    }
    if (debug_mode > 1) {
      printf("%cSaving the feature space                                            %c", 13, 13);
      fflush(stdout);
    }
    fprintf(fs, "%lld %lld\n", clcn, layer1_size);
    for (a = 0; a < clcn; a++) {
      fprintf(fs, "%lld ", a);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&fspace[a * layer1_size + b], sizeof(real), 1, fs);
      else for (b = 0; b < layer1_size; b++) fprintf(fs, "%lf ", fspace[a * layer1_size + b]);
      fprintf(fs, "\n");
    }
    fclose(fs);
  }

  fclose(ft);
  free(cl); free(feat_table);

  // Generate topics
  real *acc_vec = (real *)malloc(layer1_size * sizeof(real));
  FILE *ftr = fopen(train_file, "rb");
  FILE *fto = fopen(output_file, "wb");
  if (fto == NULL) {
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
  bool *significant_features = (bool *)malloc(clcn * sizeof(bool));
  for (a = 0; a < clcn; a++) significant_features[a] = FALSE;
  int sig_feat_cnt = 0;
  // Loop over documents in the input stream
  while(1) {
    for(a = 0; a < layer1_size; a++) acc_vec[a] = 0;
    // Loop over words in the current input document
    while(1) {
      ReadWord(word, ftr);
      // If end-of-doc or end-of-stream quit the words loop
      if ((feof(ftr)) || (strcmp (word, (char *)"</s>") == 0)) break;
      long long vocab_index = SearchVocab(word);
      //  If found add its vector to the accumulator
      if (vocab_index > -1) for(a = 0; a < layer1_size; a++) acc_vec[a] += syn0[vocab_index * layer1_size + a];
      // This is inefficient, and could be replaced with counting words and multiplying vectors by them.
      // However it does not impact the total runtime significantly, then it is left as-is for simplicity.
    }
    if (sparse) fprintf(fto, "Document %lld\n", doc_processed);
    // Normalize accumulator for the cosine distance calculation
    real len_acc = 0;
    for (a = 0; a < layer1_size; a++) len_acc += acc_vec[a] * acc_vec[a];
    len_acc = sqrt(len_acc);
    if (len_acc > 0) for (a = 0; a < layer1_size; a++) acc_vec[a] /= len_acc; // NULL document will give len_acc == 0
    // For each class, i.e. feature
    for(a = 0; a < clcn; a++) {
      // Calculate the cosine distance between the accumulator and the i-th class vector
      real x = 0;
      for (b = 0; b < layer1_size; b++) x += acc_vec[b] * fspace[a * layer1_size + b];
      // Write the feature num and the cosine distance calculated
      if (sparse) {
        if ((x > 0.3) || (x < -0.3)) fprintf(fto, " %lld %f", a, x);
      } else {
        fprintf(fto, "%f", x);
        if (a < clcn-1) fprintf(fto, ";");
      }
      if ((x < -0.5) || (x > 0.5)) significant_features[a] = TRUE;
    }
    // Display progress information
    if ((debug_mode > 1) && (doc_processed % 1000 == 0)){
      printf("%c%lldK of %lldK documents tagged                      %c", 13, doc_processed/1000, max_documents/1000, 13);
      fflush(stdout);
    }
    fprintf(fto, "\n");
    if (feof(ftr)) break;
    doc_processed++; //if (doc_processed == 100) break;
  }
  for (a = 0; a < clcn; a++) if (significant_features[a]) sig_feat_cnt++;
  if (debug_mode > 0) printf("%c%d significant features (> 0.5) found.                     \n", 13, sig_feat_cnt);
  fclose(ftr); fclose(fto); free(acc_vec); free(fspace);
}

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

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
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
    // tomekd789...
    printf("\t-k-iter <int>\n");
    printf("\t\tRun <int> K-means iterations (default 10)\n");
    printf("\t-words-per-feat <int>\n");
    printf("\t\tWrite <int> words for each feature (default 10)\n");
    printf("\t-sparse <int>\n");
    printf("\t\tProvide sparse output, cut off at 0.3; default is 0 (not used)\n");
    // ...tomekd789
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    // tomekd789, commented out:
    printf("\t-classes <int>\n");
    // tomekd789, commented out:
    //printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t\tNumber of features, i.e. K-means classes; default is equal to -size\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    // tomekd789...
    printf("\t-read-semantic-space <file>\n");
    printf("\t\tThe semantic space will be read from <file>, not computed by word2vec\n");
    printf("\t-save-semantic-space <file>\n");
    printf("\tThe semantic space will be saved to <file>\n");
    printf("\t-features <file>\n");
    printf("\tFeatures will be saved to <file> (proxy word sets)\n");
    printf("\t-f-space <file>\n");
    printf("\tThe feature space will be saved to <file> (vectors)\n");
    // ...tomekd789
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    // tomekd789, commented out:
    //printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    // tomekd789:
    printf("./doc2feat -train data.txt -output tags.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  // tomekd789...
  read_semantic_space_file[0] = 0;
  write_semantic_space_file[0] = 0;
  features_file[0] = 0;
  // ...tomekd789
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  // tomekd789...
  if ((i = ArgPos((char *)"-read-semantic-space", argc, argv)) > 0) strcpy(read_semantic_space_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-semantic-space", argc, argv)) > 0) strcpy(write_semantic_space_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-features", argc, argv)) > 0) strcpy(features_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-f-space", argc, argv)) > 0) strcpy(fs_file, argv[i + 1]);
  // ...tomekd789
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  // tomekd789...
  if ((i = ArgPos((char *)"-k-iter", argc, argv)) > 0) k_iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-words-per-feat", argc, argv)) > 0) words_per_feat = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sparse", argc, argv)) > 0) sparse = atoi(argv[i + 1]);
  // ...tomekd789
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  // tomekd789, commented out:
  // if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  // tomekd789:
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) {classes = atoi(argv[i + 1]);} else classes = layer1_size;
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  CalculateTopics();
  return 0;
}
