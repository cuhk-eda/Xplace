#include "GPUTimer.h"
#include "gputimer/db/GTDatabase.h"

using std::ofstream;
using std::string;
using std::cerr;
using std::endl;
using std::stringstream;

namespace gt {

void GPUTimer::read_spef(const std::string& file) {
    logger.info("reading spef: %s", file.c_str());
    if (not std::filesystem::exists(file)) {
        std::cerr << "can't find " << file << '\n';
        std::exit(EXIT_FAILURE);
    }

    // Invoke the read function and check the return value
    if (not spef.read(file)) {
        std::cerr << *spef.error;
        std::exit(EXIT_FAILURE);
    }


    if (spef.time_unit == "1 PS") gtdb.spef_time_unit = 1e-12;
    if (spef.time_unit == "1 NS") gtdb.spef_time_unit = 1e-9;
    if (spef.time_unit == "1 US") gtdb.spef_time_unit = 1e-6;
    if (spef.time_unit == "1 MS") gtdb.spef_time_unit = 1e-3;
    if (spef.time_unit == "1 S") gtdb.spef_time_unit = 1.0; ;

    if (spef.capacitance_unit == "1 FF") gtdb.spef_cap_unit = 1e-15;
    if (spef.capacitance_unit == "1 PF") gtdb.spef_cap_unit = 1e-12;
    if (spef.capacitance_unit == "1 NF") gtdb.spef_cap_unit = 1e-9;
    if (spef.capacitance_unit == "1 UF") gtdb.spef_cap_unit = 1e-6;
    if (spef.capacitance_unit == "1 F") gtdb.spef_cap_unit = 1.0;

    if (spef.resistance_unit == "1 OHM") gtdb.spef_res_unit = 1.0;
    if (spef.resistance_unit == "1 KOHM") gtdb.spef_res_unit = 1e3;
    if (spef.resistance_unit == "1 MOHM") gtdb.spef_res_unit = 1e6;

    logger.info("spef time_unit: %.5E s", *gtdb.spef_time_unit);
    logger.info("spef capacitance_unit: %.5E F", *gtdb.spef_cap_unit);
    logger.info("spef resistance_unit: %.5E Ohm", *gtdb.spef_res_unit);

    spef.expand_name();
}

}  // namespace gt