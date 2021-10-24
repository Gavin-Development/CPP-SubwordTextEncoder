//
// Created by Josh Shiells on 09/10/2021.
//
#include <utility>
#include <future>
#include <thread>
#include <regex>

#include "Tokenizers.h"

using namespace tokenizers;

BytePairTextEncoder::BytePairTextEncoder(unsigned long long int targetVocabSize, std::string name1,
                                         unsigned long long int target_vocab_size, std::string name)
        : SubwordTextEncoder(targetVocabSize, std::move(name1)) {
    _vocab_size = 0;
    _target_vocab_size = target_vocab_size;
    _name = std::move(name);
    _metadata["name"] = _name;
    _metadata["target_vocab_size"] = target_vocab_size;
}

std::list<std::string> BytePairTextEncoder::word_tokenize(const std::string &sentence) {
    return BytePairTextEncoder::_split(" ", sentence);
}

std::list<std::list<std::string>> BytePairTextEncoder::_batch_word_tokenize(const std::list<std::string>& texts) {
    std::list<std::list<std::string>> corpus;
    for( const auto& text : texts ) corpus.push_back(BytePairTextEncoder::word_tokenize(text));
    return corpus;
}

std::list<std::string> BytePairTextEncoder::_split(const std::string &delimiter, std::string s) {
    std::size_t pos;
    std::string token;
    std::list<std::string> values;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos) + BytePairTextEncoder::END_OF_WORD;
        values.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    values.push_back(s);
    return values;
}

WordFrequencies BytePairTextEncoder::_word_counts(const std::list<std::list<std::string>>& tokenized_text) {
    std::map<std::string, int> counter;
    for (const auto& sentence: tokenized_text) {
        for (const auto &word: sentence) {
            if (word.empty() or word == " ") continue;
            if (counter.find(word) == counter.end()) {
                counter[word] = 1;
            } else {
                counter[word] += 1;
            }
        }
    }
    return counter;
}

PairFrequencies BytePairTextEncoder::get_pair_frequency(const WordFrequencies& word_frequency) {
    PairFrequencies pair_frequency;
    for (const auto& [word, frequency]: word_frequency) {
        std::vector<char> letters(word.begin(), word.end());
        for (int i=0; i<letters.size()-1; i++) {
            std::list<std::string> letter_pair = {std::string(1, letters[i]), std::string(1, letters[i+1])};
            if (pair_frequency.find(letter_pair) == pair_frequency.end()) pair_frequency[letter_pair] = 1;
            else pair_frequency[letter_pair] += frequency;
        }
    }
    return pair_frequency;
}

WordFrequencies BytePairTextEncoder::merge_vocab(std::_Tree_iterator<std::_Tree_val<std::_Tree_simple_types<std::pair<const std::list<std::basic_string<char>>, int>>>> pair, WordFrequencies vocab_in) {
    // Regex pattern of the pair of characters
    // ?<! means "negative look behind" \S means whitespaces
    WordFrequencies vocab_out = {};
    // Build string from list.
    std::list<std::string> characters = pair->first;
    std::string pair_of_characters;
    for (const auto& character: characters) pair_of_characters += character;
    // Clean The pair of characters for use in regular expression.
    std::regex specialChars { R"([-[\]{}()*+?.,\^$|#\s\/])" };
    std::string sanitised = std::regex_replace(pair_of_characters, specialChars, R"(\$&)" );
    std::string full_pattern = "((?<!\\S)" + sanitised + "(?!\\S))";
    std::regex sub_pattern(full_pattern);

    // Substitute words & build new vocabulary.
    for (const auto& [word, frequency]: vocab_in) {
        auto words_begin = std::regex_iterator(word.begin(), word.end(), sub_pattern);
        auto words_end = std::sregex_iterator();
        std::string new_word;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            std::string match_str = match.str();
            std::string match_prefix = match.prefix();
            std::string match_suffix = match.suffix();
            new_word = match_prefix += pair_of_characters += match_suffix;
        }
        vocab_out[new_word] = vocab_in[word];
    }
    return vocab_out;
}

void BytePairTextEncoder::build_vocabulary(const std::list<std::string> &texts) {
    /* Builds the vocabulary based on the BytePairText encoder system.*/
    std::list<std::list<std::string>> chunks = BytePairTextEncoder::_chunk_corpus(texts, PROCESSOR_COUNT);
    std::vector<std::future<std::list<std::list<std::string>>>> futures;
    for (const auto& item: chunks) {
        futures.emplace_back(std::async(std::launch::async, BytePairTextEncoder::_batch_word_tokenize, item));
    }

    std::list<std::list<std::string>> corpus;
    for (auto &item: futures) {
        for (const auto& elem: item.get()) {
            corpus.push_back(elem);
        }
    }
    std::map<std::string, int> counts = BytePairTextEncoder::_word_counts(corpus);
    for (unsigned long long int i=0; i<=_target_vocab_size; i++) {
        if (counts.empty()) break;

        PairFrequencies pairs = BytePairTextEncoder::get_pair_frequency(counts);
        if (pairs.empty()) break;

        auto best = std::max_element(pairs.begin(),
                                     pairs.end(),
                                     [](const PairFrequency& a,
                                             const PairFrequency& b) -> bool {return a.second < b.second;});
        counts = BytePairTextEncoder::merge_vocab(best, counts);
    }
}
