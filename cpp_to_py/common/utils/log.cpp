#include "log.h"
#include "log_level.h"

#include <iomanip>
#include <sstream>

#if defined(__unix__)
#include <sys/resource.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

namespace utils {

timer::timer() { start(); }
void timer::start() { _start = clock::now(); }
double timer::elapsed() const { return std::chrono::duration<double>(clock::now() - _start).count(); }
std::string timer::get_time_stamp() {
    std::stringstream ss;
    ss << "[" << std::setprecision(3) << std::setw(8) << std::fixed << this->elapsed() << "] ";
    return ss.str();
}

timer tstamp;

bool verbose_parser_log = false;

PrintfLogger logger;

double mem_use::get_current() {
#if defined(__unix__)
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) {
        return 0.0; /* Can't open? */
    }
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return 0.0; /* Can't read? */
    }
    fclose(fp);
    return rss * sysconf(_SC_PAGESIZE) / 1048576.0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.WorkingSetSize / 1048576.0;
#else
    return 0.0;  // unknown
#endif
}

double mem_use::get_peak() {
#if defined(__unix__)
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return rusage.ru_maxrss / 1024.0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.PeakWorkingSetSize / 1048576.0;
#else
    return 0.0;  // unknown
#endif
}

void assert_msg(bool condition, const char* format, ...) {
    if (!condition) {
        std::cerr << "Assertion failure: ";
        va_list ap;
        va_start(ap, format);
        vfprintf(stdout, format, ap);
        std::cerr << std::endl;
        std::abort();
    }
    return;
}

std::string log_level_ANSI_color(int& log_level) {
    std::string color_string;
    switch (log_level) {
        case LOG_NOTICE:
            color_string = "\033[1;34mNotice\033[0m: ";
            break;
        case LOG_WARN:
            color_string = "\033[1;93mWarning\033[0m: ";
            break;
        case LOG_ERROR:
            color_string = "\033[1;31mError\033[0m: ";
            break;
        case LOG_FATAL:
            color_string = "\033[1;41;97m F A T A L \033[0m: ";
            break;
        case LOG_OK:
            color_string = "\033[1;32mOK\033[0m: ";
            break;
        default:
            break;
    }
    return color_string;
}


// C++ 20 only
// void PrintfLogger::setup_logger(argparse::ArgumentParser parser) {
//     std::filesystem::path current_dir(std::filesystem::current_path());
//     // std::filesystem::path result_dir(parser.get<std::string>("result_dir"));
//     std::filesystem::path result_dir(st::setting.result_dir);
//     std::filesystem::path exp_id(parser.get<std::string>("exp_id"));
//     std::filesystem::path log_dir(parser.get<std::string>("log_dir"));
//     std::filesystem::path log_name(parser.get<std::string>("log_name"));

//     std::filesystem::path res_root = current_dir / result_dir / exp_id;
//     std::filesystem::path log_root = res_root / log_dir;
//     // std::filesystem::path log_file_path = log_root / log_name;
//     std::string log_file_path = (log_root / log_name).string();

//     if (!std::filesystem::exists(log_root)) {
//         std::filesystem::create_directories(log_root);
//     }

//     if (write_log) {
//         f = fopen(log_file_path.c_str(), "w");
//         if (f == NULL) {
//             this->error("Cannot open logfile {}", log_file_path);
//             exit(1);
//         }
//     }
// }

}  // namespace utils