//
// Created by Josh Shiells on 09/10/2021.
//
#include "Tokenizers.h"

using namespace tokenizers;

std::map<char, int> BytePairTextEncoder::_word_counts(const std::list<std::list<std::string>> &tokenized_text) {
    std::map<char, int> counts;
    for (const auto& sentence: tokenized_text) {
        for (const auto& word: sentence) {
            for (const char& letter: word) {
                auto iter = counts.find(letter);
                if (iter == counts.end()) {
                    counts[letter] = 1;
                }
                else {
                    counts[letter] = counts[letter] + 1;
                }
            }
        }
    }
    return counts;
}
