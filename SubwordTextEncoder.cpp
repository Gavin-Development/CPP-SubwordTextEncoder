#include "Tokenizers.h"
#include <map>
#include <list>
#include <regex>
#include <fstream>
#include <thread>
#include <future>

using json = nlohmann::json; // For convince sake.
using namespace tokenizers;

SubwordTextEncoder::SubwordTextEncoder(unsigned long long int target_vocab_size, std::string name) {
    _vocab_size = 0;
    _target_vocab_size = target_vocab_size;
    _name = std::move(name);
    _metadata["name"] = _name;
    _metadata["target_vocab_size"] = target_vocab_size;
}

SubwordTextEncoder::SubwordTextEncoder(const std::string& filename_prefix) {
    _vocab_size = 0;
    std::string filename = SubwordTextEncoder::filename(filename_prefix);
    std::ifstream SubwordFile(filename);
    int line=0;
    std::string buffer;
    while (std::getline(SubwordFile, buffer))
    {
        ++line;
        if (line == 1) {continue;}
        else if (line == 2) {
            std::string unparsed = buffer.substr(buffer.find(METADATA_PREFIX) + METADATA_PREFIX.size());
            json metadata = json::parse(unparsed);
            if (not metadata.empty()) {
                _target_vocab_size = metadata["target_vocab_size"];
            _name = metadata["name"];
            _metadata = metadata;}
        }
        else if (not buffer.empty()) {
            std::string word = buffer.substr(1, buffer.size()-2);
            vocabulary.push_back(word);
            _subword_to_id[word] = _vocab_size;
            ++_vocab_size;
        }
    }
}

std::map<std::string, int> SubwordTextEncoder::_word_counts(const std::list<std::list<std::string>>& tokenized_text) {
    std::map<std::string, int> counter;
    for (const auto& sentence: tokenized_text) {
        for (const auto &word: sentence) {
            if (counter.find(word) == counter.end()) {
                counter[word] = 1;
            } else {
                counter[word] += 1;
            }
        }
    }
    return counter;
}

std::string SubwordTextEncoder::space_punctuation(const std::string& s) {
    std::regex regexp("([?.!,'*\\\":@])");
    return std::regex_replace(s, regexp, " $1");
}

std::list<std::string> SubwordTextEncoder::word_tokenize(const std::string& text) {
    return SubwordTextEncoder::_split(" ", std::move(space_punctuation(text)));
}

std::string SubwordTextEncoder::_split_word(std::string s) {
    std::string split_word;
    for (char& c : s) {
        if (c != s.back()) {split_word + c + " ";}
        else {split_word + c;}

    }
    return split_word + " " + END_OF_WORD;
}

std::list<std::string> SubwordTextEncoder::_split(const std::string &delimiter, std::string s) {
        std::size_t pos = 0;
        std::string token;
        std::list<std::string> values;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            values.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        values.push_back(s);
        return values;
}

std::list<std::list<std::string>> SubwordTextEncoder::_chunk_corpus(std::list<std::string> texts, unsigned int n) {
    std::list<std::list<std::string>> out;
    unsigned long long int max_size = texts.size() / n;
    auto texts_it = texts.begin();
    for (int i=0; i<n; ++i) {
        std::list<std::string> temp_out;
        for (int x=0; x<max_size; ++x) {
            temp_out.push_back(*texts_it);
            ++texts_it;
        }
        out.push_back(temp_out);
    }
    return out;
}

std::list<std::list<std::string>> SubwordTextEncoder::_batch_word_tokenize(const std::list<std::string>& texts) {
    std::list<std::list<std::string>> corpus;
    for( const auto& text : texts ) corpus.push_back(SubwordTextEncoder::word_tokenize(text));
    return corpus;
}

void SubwordTextEncoder::build_vocabulary(const std::list<std::string>& texts) {
    /* Build the vocabulary out of the most common words in the texts,
        Keep adding values to the vocab till target vocabulary length is reached
        or we go down to counts of "1".*/

    std::list<std::list<std::string>> chunks = SubwordTextEncoder::_chunk_corpus(texts, PROCESSOR_COUNT);
    std::vector<std::future<std::list<std::list<std::string>>>> futures;
    for (const auto& item: chunks) {
        futures.emplace_back(std::async(std::launch::async, SubwordTextEncoder::_batch_word_tokenize, item));
    }
    std::list<std::list<std::string>> corpus;
    for (auto &item: futures) {
        for (const auto& elem: item.get()) {
            corpus.push_back(elem);
        }
    }
    std::map<std::string, int> counts = SubwordTextEncoder::_word_counts(corpus);
    if (vocabulary.empty()) {
        for (unsigned long long int i=0; i < _target_vocab_size; i++) {
            if( counts.empty() ) break;

            auto best = std::max_element(counts.begin(),
                                         counts.end(),
                                         [](const std::pair<std::string, int> &a,
                                            const std::pair<std::string, int> &b) -> bool {return a.second < b.second;});
            if( best->second <= 1 ) break;
            vocabulary.push_back(best->first);
            _subword_to_id[best->first] = _vocab_size;
            counts.erase(best->first);
            if (best->first.length() > largest_word) largest_word = best->first.length();
            if (best->first.length() < smallest_word) smallest_word = best->first.length();
            ++_vocab_size;
        }
    }
    else {
        for (unsigned long long int i=vocabulary.size(); i < _target_vocab_size; i++) {
            if (counts.empty()) break;
            auto best = std::max_element(counts.begin(),
                                         counts.end(),
                                         [](const std::pair<std::string, int> &a,
                                            const std::pair<std::string, int> &b) -> bool {return a.second < b.second;});
            if( best->second <= 1 ) break;
            if (std::find(vocabulary.begin(), vocabulary.end(), best->first) != vocabulary.end()) continue;
            vocabulary.push_back(best->first);
            _subword_to_id[best->first] = _vocab_size;
            counts.erase(best->first);
            ++_vocab_size;
        }
    }
}

void SubwordTextEncoder::write_lines(const std::string& filename_prefix) {
    std::string filename = SubwordTextEncoder::filename(filename_prefix);
    std::ofstream SubwordFile(filename);
    SubwordFile << HEADER_PREFIX << _name << "\n";
    if (_metadata.dump() != "null") SubwordFile << METADATA_PREFIX << _metadata.dump() << "\n";
    else SubwordFile << METADATA_PREFIX << "{}";
    for (const auto& item: vocabulary) {
        SubwordFile << "'" << item << "'" << "\n";
    }
    SubwordFile.close();
}

std::list<int> SubwordTextEncoder::_byte_encode(const std::string& token) const {
    unsigned long long int offset = vocabulary.size();
    std::list<int> out;
    for (char const& c: token) {
        out.push_back(c+offset);
    }
    return out;
}

std::list<int> SubwordTextEncoder::encode(const std::string& sentence) {
    std::list<int> out;
    std::list<std::string> tokenized_word = SubwordTextEncoder::word_tokenize(sentence);
    for (auto& item: tokenized_word) {
        if (_subword_to_id.find(item) != _subword_to_id.end()) out.push_back(_subword_to_id[item]);
        else {
            for (auto &byte: _byte_encode(item)) {
                out.push_back(byte);
            }
        }
    }
    return SubwordTextEncoder::_pad_incr(out);
}

std::string SubwordTextEncoder::_id_to_subword(int id) {
    if (id < _vocab_size) {
        auto vocab_it = vocabulary.begin();
        std::advance(vocab_it, id);
        return *vocab_it;
    }
    else {
        unsigned long long int offset = vocabulary.size();
        std::string out;
        char c = id-offset;
        return out + c;
    }
}

std::list<int> SubwordTextEncoder::_pad_incr(const std::list<int>& ids) {
    std::list<int> out;
    for (auto& item: ids) {out.push_back(item+1);}
    return out;
}

std::string SubwordTextEncoder::decode(const std::list<int>& sentence_encoded) {
    std::string out;
    for (auto& id: sentence_encoded) {
        out += SubwordTextEncoder::_id_to_subword(id - 1);
    }
    return out;
}
