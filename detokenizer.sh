#!/bin/bash -v
# Usage - ./detokenizer.sh predictions target_lang

mosesdecoder=/nas/home/jwei/mosesdecoder

predictions=$1
target=$2

# Get BLEU Score
cat $predictions \
    | sed 's/\@\@ //g' \
    | sed 's/<EOS>.*//g' \
    | $mosesdecoder/scripts/recaser/detruecase.perl \
    | $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $target \
    > $predictions.detok
