#include "SubwordTextEncoder.h"
#include <vector>
#include <fstream>
#include <iostream>

std::list<std::string> load_text() {
    std::list<std::string> texts;
    std::ifstream input("/home/scot/Downloads/readme.txt");
    std::string lineBuffer;
    while (std::getline(input, lineBuffer)) {
        texts.push_back(lineBuffer);
    }
    return texts;
}

void print_table(std::list<std::string> vocab_list, unsigned long int word_size) {
    auto vocab_front = vocab_list.cbegin();
    for (int i=0; i < 8 + word_size; i++) std::cout << "-";
    std::cout << std::endl;
    for (int i=0; i < 5; i++) {
        std::cout << "| " << i << " | " << *vocab_front;
        for (int ll=0; ll < word_size-vocab_front->length(); ll++) std::cout << " ";
        std::cout << " |" << std::endl;
        std::advance(vocab_front, i);
    }
    for (int i=0; i < 8 + word_size; i++) std::cout << "-";
    std::cout << std::endl;
}

int main() {
    tokenizers::SubwordTextEncoder TextEncoder(1000, "Test");
    std::list<std::string> texts = load_text();
    auto l_front = texts.cbegin();
    std::cout << "Example Of Vocab: " << *l_front << std::endl;
    TextEncoder.build_vocabulary(texts);
    std::list<std::string> vocab = TextEncoder.get_vocabulary();
    std::cout << "Top 5 Vocabulary: " << std::endl;
    std::cout << "Size of word " << TextEncoder.largest_word << std::endl;
    print_table(vocab, TextEncoder.largest_word);
    return 0;
}