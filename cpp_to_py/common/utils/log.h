//
// Some logging utilities
// 1. "log() << ..." will show a time stamp first
// 2. "print(a, b, c)" is python-like print for any a, b, c that has operator<< overloaded. For example,
//      int a = 10;
//      double b = 3.14;
//      std::string c = "Gengjie";
//      print(a, b, c);
//     This code piece will show "10 3.14 Gengjie".
// 3. "assert_msg(condition, format, ...)" is python-like assert
//

#pragma once

#include <chrono>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "log_level.h"
// #include "argparse.hpp"  // need C++20

namespace utils {

extern bool verbose_parser_log;

// 1. Timer


class timer {
    using clock = std::chrono::high_resolution_clock;

private:
    clock::time_point _start;

public:
    timer();
    void start();
    double elapsed() const;  // seconds
    std::string get_time_stamp();
};

extern timer tstamp;

// 2. Memory

class mem_use {
public:
    static double get_current();  // MB
    static double get_peak();     // MB
};

// 3. Easy print

// print(a, b, c)
inline void print() { std::cout << std::endl; }
template <typename T, typename... TAIL>
void print(const T& t, TAIL... tail) {
    std::cout << t << ' ';
    print(tail...);
}

// "printlog(LOG_LEVEL, a, b, c...)" puts a time stamp in beginning
// try to make code compatible with old printlog(int level, const char *format, ...)
void printlog(int level, const char* format, ...);

void assert_msg(bool condition, const char* format, ...);

std::string log_level_ANSI_color(int& log_level);

// 4. PrintfLogger format and write to file
// Python kind logger, support printf kind format (can be replaced by fmt library)
class PrintfLogger {
    static constexpr bool write_log = false;
    FILE* f;

public:
    // void setup_logger(argparse::ArgumentParser parser);
    ~PrintfLogger() {
        if (f != NULL) fclose(f);
    }

    template <typename... Args>
    void log(int log_level, const char* format, Args&&... args) {
        if (!verbose_parser_log) {
            return;
        }
        if (log_level >= GLOBAL_LOG_LEVEL) {
            std::string curr_log = tstamp.get_time_stamp();
            if (log_level > LOG_INFO) {
                curr_log += log_level_ANSI_color(log_level);
            }
            std::cout << curr_log;
            printf(format, args...);
            printf("\n");
            fflush(stdout);
            if (write_log) {
                fprintf(f, "%s", curr_log.c_str());
                fprintf(f, format, args...);
                fprintf(f, "\n");
                fflush(f);
            }
        }
    }

    template <typename... Args>
    void debug(const char* format, Args&&... args) {
        log(LOG_DEBUG, format, args...);
    };
    template <typename... Args>
    void verbose(const char* format, Args&&... args) {
        log(LOG_VERBOSE, format, args...);
    };
    template <typename... Args>
    void info(const char* format, Args&&... args) {
        log(LOG_INFO, format, args...);
    };
    template <typename... Args>
    void notice(const char* format, Args&&... args) {
        log(LOG_NOTICE, format, args...);
    };
    template <typename... Args>
    void warning(const char* format, Args&&... args) {
        log(LOG_WARN, format, args...);
    };
    template <typename... Args>
    void error(const char* format, Args&&... args) {
        log(LOG_ERROR, format, args...);
    };
    template <typename... Args>
    void fatal(const char* format, Args&&... args) {
        log(LOG_FATAL, format, args...);
    };
    template <typename... Args>
    void ok(const char* format, Args&&... args) {
        log(LOG_OK, format, args...);
    };
};

extern PrintfLogger logger;

}  // namespace utils