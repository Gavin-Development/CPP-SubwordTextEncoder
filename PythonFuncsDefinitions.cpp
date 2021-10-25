//
// Created by Josh Shiells on 25/10/2021.
//
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Tokenizers.h"

namespace py = pybind11;
using namespace tokenizers;

int InitModule() {
    std::cout << "Tokenizer library accelerated with C++\n";
    return 0;
}

PYBIND11_MODULE(GavinTokenizers, handle) {
    InitModule();
    handle.doc() = "This module is a custom module written in c++ to accelerate tokenizing loading for Gavin made by Josh (Scot_Survivor)";
    py::class_<SubwordTextEncoder>(handle, "SubwordTextEncoder")
            .def(py::init<unsigned long long int &, std::string &>())
            .def(py::init<const std::string &>())
            .def("word_tokenize", &SubwordTextEncoder::word_tokenize)
            .def("get_vocab_size", &SubwordTextEncoder::get_vocab_size)
            .def("get_vocabulary", &SubwordTextEncoder::get_vocabulary)
            .def("get_name", &SubwordTextEncoder::get_name)
            .def("build_vocabulary", &SubwordTextEncoder::build_vocabulary)
            .def("write_lines", &SubwordTextEncoder::write_lines)
            .def("filename", &SubwordTextEncoder::filename)
            .def("encode", &SubwordTextEncoder::encode)
            .def("decode", &SubwordTextEncoder::decode)
            .def("__repr__",
                 [](const SubwordTextEncoder &a) {
                    return "<GavinTokenizers.SubwordTextEncoder name '" + a.get_name() + "'>";});

    #ifdef VERSION_INFO
        handle.attr("__version__") = VERSION_INFO;
    #else
        handle.attr("__version__") = "dev";
    #endif
}


