#include "BindHelper.h"

namespace Xplace {

bool loadParams(const py::dict& kwargs) {
    db::setting.reset();
    // ----- design related options -----

    if (kwargs.contains("bookshelf_variety")) {
        db::setting.BookshelfVariety = kwargs["bookshelf_variety"].cast<std::string>();
    }

    if (kwargs.contains("aux")) {
        db::setting.BookshelfAux = kwargs["aux"].cast<std::string>();
    }

    if (kwargs.contains("pl")) {
        db::setting.BookshelfPl = kwargs["pl"].cast<std::string>();
    }

    if (kwargs.contains("def")) {
        db::setting.DefFile = kwargs["def"].cast<std::string>();
    }

    if (kwargs.contains("lef")) {
        db::setting.LefFile = kwargs["lef"].cast<std::string>();
    } else if (kwargs.contains("cell_lef") && kwargs.contains("tech_lef")) {
        db::setting.LefCell = kwargs["cell_lef"].cast<std::string>();
        db::setting.LefTech = kwargs["tech_lef"].cast<std::string>();
    }

    if (kwargs.contains("constraints")) {
        db::setting.Constraints = kwargs["constraints"].cast<std::string>();
    }

    if (kwargs.contains("output")) {
        db::setting.OutputFile = kwargs["output"].cast<std::string>();
    }

    if (db::setting.BookshelfAux == "" && db::setting.DefFile == "") {
        logger.error("design is not found");
        return false;
    }

    // verilog is unused now
    // if (kwargs.contains("verilog")) {
    //     db::setting.Verilog = kwargs["verilog"].cast<std::string>();
    // }

    // ----- other options -----

    // db loading mode
    if (kwargs.contains("lite_mode")) {
        db::setting.liteMode = kwargs["lite_mode"].cast<bool>();
    }

    // enable random place or not
    if (kwargs.contains("random_place")) {
        db::setting.random_place = kwargs["random_place"].cast<bool>();
    }

    // verbose on/off in parser, default is verbose off(false)
    if (kwargs.contains("verbose_parser_log")) {
        utils::verbose_parser_log = kwargs["verbose_parser_log"].cast<bool>();
    } else {
        utils::verbose_parser_log = false;
    }

    if (kwargs.contains("num_threads")) {
        db::setting.numThreads = kwargs["num_threads"].cast<int>();
    }

    return true;
}

std::tuple<std::shared_ptr<db::Database>, std::shared_ptr<gp::GPDatabase>> start_all(const py::dict& kwargs) {
    bool load_status = loadParams(kwargs);
    if (!load_status) {
        throw std::invalid_argument("Received invalid params. Please check!");
    }
    auto rawdb_ptr = std::make_shared<db::Database>();
    rawdb_ptr->load();
    rawdb_ptr->setup();
    auto gpdb_ptr = std::make_shared<gp::GPDatabase>(rawdb_ptr);
    gpdb_ptr->setup();
    return {rawdb_ptr, gpdb_ptr};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<db::Database, std::shared_ptr<db::Database>>(m, "Database")
        .def(pybind11::init<>())
        .def("load", &db::Database::load)
        .def("setup", &db::Database::setup)
        .def("reset", &db::Database::reset);
    bindGPDatabase(m);
    m.def("load_params", &loadParams, "Parse input args to DB and return graph information");
    m.def("create_database", []() { return std::make_shared<db::Database>(); });
    m.def("create_gpdatabase", [](std::shared_ptr<db::Database> db) { return std::make_shared<gp::GPDatabase>(db); });
    m.def("start", &start_all, "Parse input args to DB and return pointer from raw db and gp db");
}

}  // namespace Xplace
