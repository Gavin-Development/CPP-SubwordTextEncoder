#include "SubwordTextEncoder.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>

struct CommandLineArgs {
    int handle;
    unsigned long long int vocab_size;
    std::string filename;
};

std::list<std::string> load_text(std::string filename) {
    std::list<std::string> texts;
    std::ifstream input(filename);
    std::string lineBuffer;
    while (std::getline(input, lineBuffer)) {
        texts.push_back(lineBuffer);
    }
    return texts;
};

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
};

CommandLineArgs handle_cli(int argc, char *argv[]) {
    unsigned long long int vocab_size = 0;
    std::string filename = "./readme.txt";
    int handle = 0;
    if (argc == 1) return CommandLineArgs{handle, vocab_size, filename};
    if (argc > 1) {
        std::string argument = argv[1];
        if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: TokenizerTest [options]" << std::endl;
            std::cout << "  -h/--help: Displays this help." << std::endl;
            std::cout << "  -i/--input-corpus: Set a filename for a corpus." << std::endl;
            handle = -1;
        }
        for (int ii=0; ii<argc; ii++) {
            std::string argument = argv[ii];
           if (argument == "--input-corpus" || argument == "-i") {
                handle = 1;
                filename = argv[ii+1];
            }
        }
        if (handle == 0) {
            std::cout << "Unknown ";
            if (argc == 2) {
                std::cout << "argument ";
                for (int ii=1; ii<argc-1; ii++) {
                std::string argument = argv[ii];
                if (argument != "--input-corpus" || argument != "-i") {
                    std::cout << argument;
                    if (argc-1 != ii) {
                        std::cout << " ";
                    }
                } 
            }
            }
            else if (argc > 2) {
                std::cout << "arguments ";
                for (int ii=1; ii<argc; ii++) {
                std::string argument = argv[ii];
                if (argument != "--input-corpus" || argument != "-i") {
                    std::cout << argument;
                    if (argc-1 != ii) {
                        std::cout << " ";
                    }
                } 
            }
            }
            std::cout << std::endl;
            handle = -1;
        }
    }
    return CommandLineArgs{handle, vocab_size, filename};
};

int main(int argc, char *argv[]) {
    CommandLineArgs ArgsCodes = handle_cli(argc, argv);
    int handle = ArgsCodes.handle;
    unsigned long long int vocab_size = ArgsCodes.vocab_size;
    std::string filename = ArgsCodes.filename;
    if (handle == 0 || handle == 1) {
        tokenizers::SubwordTextEncoder TextEncoder(1000, "Test");
        std::list<std::string> texts = load_text(filename);
        auto l_front = texts.cbegin();
        std::cout << "Example Of Vocab: " << *l_front << std::endl;
        TextEncoder.build_vocabulary(texts);
        std::list<std::string> vocab = TextEncoder.get_vocabulary();
        std::cout << "Top 5 Vocabulary: " << std::endl;
        std::cout << "Size of largest word " << TextEncoder.largest_word << std::endl;
        std::cout << "Size of vocabulary " << TextEncoder.get_vocab_size() << std::endl;
        print_table(vocab, TextEncoder.largest_word);
        std::list<int> encoded_words = TextEncoder.encode("Hello");
        std::cout << "Hello encoded ";
        for (auto &word: encoded_words) {
            std::cout << word << " ";
        }
        std::cout << std::endl;
    }
    else if (handle == 2){
        tokenizers::SubwordTextEncoder TextEncoder(vocab_size, "Test");
        std::list<std::string> texts = load_text(filename);
        auto l_front = texts.cbegin();
        std::cout << "Example Of Vocab: " << *l_front << std::endl;
        TextEncoder.build_vocabulary(texts);
        std::list<std::string> vocab = TextEncoder.get_vocabulary();
        std::cout << "Top 5 Vocabulary: " << std::endl;
        std::cout << "Size of largest word " << TextEncoder.largest_word << std::endl;
        std::cout << "Size of vocabulary " << TextEncoder.get_vocab_size() << std::endl;
        print_table(vocab, TextEncoder.largest_word);
    }
    return 0;
};