#include <gtest/gtest.h>
#include <fstream>
#include "SubwordTextEncoder.h"

std::list<std::string> load_text(std::string filename) {
    std::list<std::string> texts;
    std::ifstream input(filename);
    std::string lineBuffer;
    while (std::getline(input, lineBuffer)) {
        texts.push_back(lineBuffer);
    }
    return texts;
};

class SubwordTextEncoderTest : public ::testing::Test {
    protected:
    void SetUp() override {
        vocab = load_text("./readme.txt");
        TextEncoderVocabFilled.build_vocabulary(vocab);
    }

    std::list<std::string> vocab;
    std::string name = "Test";
    int vocab_size = 1000;
    tokenizers::SubwordTextEncoder TextEncoderVocabEmpty = tokenizers::SubwordTextEncoder(vocab_size, name);
    tokenizers::SubwordTextEncoder TextEncoderVocabFilled = tokenizers::SubwordTextEncoder(vocab_size, name);
    std::list<int> hello_encoded{1073, 1102, 1109, 1109, 1112};
    std::string hello_decoded = "Hello";
};

TEST_F(SubwordTextEncoderTest, VocabEmptyOnInitialisation) {
    EXPECT_EQ(TextEncoderVocabEmpty.get_vocab_size(), 0) << "Vocabulary size should be 0 on initilistation.";
}

TEST_F(SubwordTextEncoderTest, NameIsInitilised) {
    EXPECT_EQ(TextEncoderVocabEmpty.get_name(), name) << "Name is not initilised correctly.";
}

TEST_F(SubwordTextEncoderTest, VocabsizeIsInitilised) {
    EXPECT_EQ(TextEncoderVocabEmpty.get_vocab_size(), 0) << "Vocab Size is not initilised correctly.";
}

TEST_F(SubwordTextEncoderTest, VocabBuilds) {
    EXPECT_EQ(TextEncoderVocabFilled.get_vocabulary().size(), vocab_size) << "Vocabulary is not building.";
}

TEST_F(SubwordTextEncoderTest, EncodesHello) {
    ASSERT_EQ(TextEncoderVocabFilled.encode("Hello").size(), 5) << "Encoding did not work. Size different than expected.";
    EXPECT_EQ(TextEncoderVocabFilled.encode("Hello"), hello_encoded) << "Encoding is not correct. Do you need to update the test phrase?";
}

TEST_F(SubwordTextEncoderTest, DecodesHello) {
    ASSERT_EQ(TextEncoderVocabFilled.decode(hello_encoded).length(), hello_decoded.length()) << "Decoding did not work. Size different to expected value. Returned value: " << TextEncoderVocabFilled.decode(hello_encoded) << " Expected Length: " << hello_decoded;
    EXPECT_EQ(TextEncoderVocabFilled.decode(hello_encoded), hello_decoded) << "Decoding is not correct. Do you need to update the test phrase?";
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}